import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, ViTModel, T5ForConditionalGeneration
from torchvision import transforms
from PIL import Image
import os
from transformers import get_cosine_schedule_with_warmup, AdamW
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist

# def setup_ddp():
#     print("Initializing Process Group...")
#     dist.init_process_group(backend='nccl')
#     local_rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     print(f"Process group initialized: Rank {local_rank}/{world_size - 1}")
#     torch.cuda.set_device(local_rank)
#     device = torch.device('cuda', local_rank)
#     return device

class QuestionEncoding(nn.Module):
    def __init__(self, pretrained_model):
        super(QuestionEncoding, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model)
        self.hidden_dim = self.encoder.config.hidden_size
        self.projection_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.encoder.config.num_layers)])

    def forward(self, question):
        input_ids = question["input_ids"]
        attention_mask = question["attention_mask"]
        encoded_question = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        projected_question = [projection(encoded_question) for projection in self.projection_layers]
        return projected_question
    
class QuestionFusing(nn.Module):
    def __init__(self, hidden_dim):
        super(QuestionFusing, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.gating_projection = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, visual_features, question_features):
        batch_size, _ = visual_features.size()
        visual_features = visual_features.unsqueeze(1)
        fused_features = torch.cat((visual_features, question_features), dim=1)
        attention_output, _ = self.attention(fused_features, fused_features, fused_features)
        visual_output = attention_output[:, 0, :].unsqueeze(1)
        projected_output = self.projection(visual_output)
        gated_output = self.gating_projection(visual_output) * torch.tanh(self.beta)
        fused_output = projected_output + gated_output
        fused_output = fused_output.squeeze(1)
        return fused_output
    
class QAViT(nn.Module):
    def __init__(self, vision_model, pretrained_model, fusion_layers):
        super(QAViT, self).__init__()
        self.vision_model = vision_model
        self.question_encoding = QuestionEncoding(pretrained_model)
        self.question_fusing = nn.ModuleList([QuestionFusing(self.question_encoding.hidden_dim) for _ in range(fusion_layers)])
        self.fusion_layers = fusion_layers

    def forward(self, image, question):
        visual_outputs = self.vision_model(pixel_values=image)
        visual_features = visual_outputs.last_hidden_state
        question_features = self.question_encoding(question)

        num_layers = visual_features.shape[1]
        start_layer = num_layers - self.fusion_layers

        for i in range(start_layer, num_layers):
            visual_features[:, i, :] = self.question_fusing[i - start_layer](visual_features[:, i, :], question_features[i - start_layer])

        visual_outputs.last_hidden_state = visual_features
        return visual_outputs
    
# Define the training dataset
class QADataset(Dataset):
    def __init__(self, image_dir, annotations, tokenizer, transform):
        self.image_dir = image_dir
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation["image_id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        question = annotation["question"]
        answer = annotation["multiple_choice_answer"]

        question_tokens = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True)

        return {
            "image": image,
            "question_tokens": {k: v.squeeze(0) for k, v in question_tokens.items()},
            "answer": answer
        }
        
def qa_collate_fn(batch):
    # Stack images
    images = torch.stack([item["image"] for item in batch])

    # Prepare questions
    question_tokens = {k: torch.cat([item["question_tokens"][k] for item in batch]) for k in batch[0]["question_tokens"]}
    padded_question_tokens = {
        "input_ids": torch.nn.utils.rnn.pad_sequence([item["question_tokens"]["input_ids"] for item in batch], batch_first=True, padding_value=0),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([item["question_tokens"]["attention_mask"] for item in batch], batch_first=True, padding_value=0),
    }

    # Tokenize and pad answers
    answers = [item["answer"] for item in batch]
    answer_tokens = tokenizer(answers, return_tensors="pt", padding=True, truncation=True)

    return {
        "image": images,
        "question_tokens": padded_question_tokens,
        "answer": answer_tokens
    }


def load_and_split_dataset(image_dir, json_path, tokenizer, transform):
    # Load the annotations from JSON
    with open(json_path, 'r') as file:
        data = json.load(file)
    annotations = data["annotations"]

    # Split into train (80%), val (10%), and test (10%)
    train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)
    val_annotations, test_annotations = train_test_split(test_annotations, test_size=0.5, random_state=42)

    # Create Dataset objects
    train_dataset = QADataset(image_dir, train_annotations, tokenizer, transform)
    val_dataset = QADataset(image_dir, val_annotations, tokenizer, transform)
    test_dataset = QADataset(image_dir, test_annotations, tokenizer, transform)

    return train_dataset, val_dataset, test_dataset

# Apply LoRa to T5 model
def apply_lora_to_t5(t5_model):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "k"]
    )
    return get_peft_model(t5_model, lora_config)

def train_qavit_with_validation(qavit_model, t5_model, train_loader, val_loader, tokenizer, num_epochs, device):
    qavit_model.train()
    t5_model.train()

    # Apply LoRa to T5 model
    t5_model = apply_lora_to_t5(t5_model)
    
    # t5_model = nn.DataParallel(t5_model)

    # Optimizer and Scheduler
    optimizer = AdamW([
        {"params": qavit_model.parameters(), "lr": 1e-4},
        {"params": t5_model.parameters(), "lr": 5e-5}
    ])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=len(train_loader) * num_epochs
    )
    
    # local_rank = torch.distributed.get_rank()  # Get local rank

    for epoch in range(num_epochs):
        # Training
        total_loss = 0
        qavit_model.train()
        t5_model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            question_tokens = {k: v.to(device) for k, v in batch["question_tokens"].items()}
            answers = {k: v.to(device) for k, v in batch["answer"].items()}

            optimizer.zero_grad()

            # Forward pass through QA-ViT model
            visual_outputs = qavit_model(images, question_tokens)
            visual_features = visual_outputs.last_hidden_state.mean(dim=1)

            # Forward pass through T5 model
            encoder_outputs = (visual_features.unsqueeze(1).repeat(1, question_tokens["input_ids"].size(1), 1), None, None)

            t5_outputs = t5_model(
                input_ids=question_tokens["input_ids"],
                attention_mask=question_tokens["attention_mask"],
                encoder_outputs=encoder_outputs,
                labels=answers["input_ids"]
            )

            loss = t5_outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_dir = f"/work/pi_dhruveshpate_umass_edu/dpadalia/models/epoch_{epoch+1}"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(qavit_model.state_dict(), f"{model_dir}/qavit_model_epoch_{epoch+1}.pth")
            t5_model.save_pretrained(f"{model_dir}/t5_model_epoch_{epoch+1}")
        
        # Validation
        qavit_model.eval()
        t5_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                question_tokens = {k: v.to(device) for k, v in batch["question_tokens"].items()}
                answers = {k: v.to(device) for k, v in batch["answer"].items()}

                # Forward pass through QA-ViT model
                visual_outputs = qavit_model(images, question_tokens)
                visual_features = visual_outputs.last_hidden_state.mean(dim=1)

                # Forward pass through T5 model
                encoder_outputs = (visual_features.unsqueeze(1).repeat(1, question_tokens["input_ids"].size(1), 1), None, None)

                t5_outputs = t5_model(
                    input_ids=question_tokens["input_ids"],
                    attention_mask=question_tokens["attention_mask"],
                    encoder_outputs=encoder_outputs,
                    labels=answers["input_ids"]
                )

                loss = t5_outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    # torch.save(qavit_model.state_dict(), "/work/pi_dhruveshpate_umass_edu/dpadalia/models/vit_base_flant5_base_05_08/qavit_model.pth")
    # t5_model.save_pretrained("/work/pi_dhruveshpate_umass_edu/dpadalia/models/vit_base_flant5_base_05_08/t5_model.pth")
    

# Define the data paths and questions/answers
image_dir = "/work/pi_dhruveshpate_umass_edu/dpadalia/dataset/combined/"
json_path = "/work/pi_dhruveshpate_umass_edu/dpadalia/dataset/filtered_combine.json"

# Load and preprocess the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tokenizer and model initialization
pretrained_model = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(pretrained_model)

# Create the dataset and dataloader
train_dataset, val_dataset, test_dataset = load_and_split_dataset(image_dir, json_path, tokenizer, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=qa_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=qa_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=qa_collate_fn)

# Load the ViT base model
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Initialize the QA-ViT model
fusion_layers = 4
qavit_model = QAViT(vit_model, pretrained_model, fusion_layers)

# Load the T5 model for conditional generation
t5_model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

# Move models to the appropriate device (e.g., GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = setup_ddp()

# qavit_model = nn.DataParallel(qavit_model)
# t5_model = nn.DataParallel(t5_model)

qavit_model.to(device)
t5_model.to(device)

# Train the QA-ViT model
num_epochs = 200
train_qavit_with_validation(qavit_model, t5_model, train_loader, val_loader, tokenizer, num_epochs, device)