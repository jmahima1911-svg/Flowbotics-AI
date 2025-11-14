"""
ppo_training_pipeline.py
Complete PPO Training Pipeline using Automated RLHF Feedback
Uses TRL (Transformer Reinforcement Learning) library
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import json
from datasets import Dataset
from RLHF_automated_feedback import AutomatedRLHFFeedbackSystem


class PPOTrainingPipeline:
    def __init__(
        self,
        model_name="meta-llama/Llama-2-7b-chat-hf",  # or your model
        reward_model_name="facebook/roberta-hate-speech-dynabench-r4-target",
        output_dir="./ppo_trained_model"
    ):
        """Initialize PPO training pipeline"""
        
        print("="*80)
        print("PPO TRAINING PIPELINE - AUTOMATED RLHF")
        print("="*80 + "\n")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}\n")
        
        # Load base model and tokenizer
        print("Loading base model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.to(self.device)
        
        print("✓ Base model loaded\n")
        
        # Load reward model (RoBERTa toxicity detector)
        print("Loading reward model...")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name
        )
        self.reward_model.to(self.device)
        self.reward_model.eval()
        
        print("✓ Reward model loaded\n")
        
        # Initialize RLHF system
        self.rlhf = AutomatedRLHFFeedbackSystem()
        
        # Output directory
        self.output_dir = output_dir
        
        # PPO Configuration
        self.ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=1.41e-5,
            batch_size=16,
            mini_batch_size=4,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            ppo_epochs=4,
            seed=0,
        )
        
        print("✓ PPO configuration set")
        print(f"  - Learning rate: {self.ppo_config.learning_rate}")
        print(f"  - Batch size: {self.ppo_config.batch_size}")
        print(f"  - PPO epochs: {self.ppo_config.ppo_epochs}\n")
    
    def compute_reward(self, response_text):
        """
        Compute reward using RoBERTa toxicity model
        Lower toxicity = Higher reward
        """
        
        # Tokenize
        inputs = self.reward_tokenizer(
            response_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get toxicity prediction
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        # Get hate probability (toxicity score)
        hate_prob = predictions[0][0].item()
        
        # Convert to reward (invert toxicity)
        # Low toxicity (0.0) -> High reward (1.0)
        # High toxicity (1.0) -> Low reward (-1.0)
        reward = 1.0 - (2.0 * hate_prob)
        
        return torch.tensor([reward])
    
    def prepare_dataset(self, training_data_file=None):
        """
        Prepare dataset for PPO training
        Can use collected feedback or custom data
        """
        
        if training_data_file:
            # Load from file
            with open(training_data_file, 'r') as f:
                data = json.load(f)
        else:
            # Use collected RLHF data
            data = self.rlhf.get_reward_dataset()
        
        if not data:
            raise ValueError("No training data available!")
        
        # Format dataset
        formatted_data = {
            'query': [item['question'] for item in data],
            'response': [item['response'] for item in data]
        }
        
        dataset = Dataset.from_dict(formatted_data)
        
        print(f"✓ Dataset prepared with {len(dataset)} examples\n")
        return dataset
    
    def train(self, dataset, num_epochs=1, save_freq=100):
        """
        Train model using PPO with automated toxicity rewards
        """
        
        print("="*80)
        print("STARTING PPO TRAINING")
        print("="*80 + "\n")
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 128,
        }
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'='*80}\n")
            
            for i, batch in enumerate(dataset):
                query = batch['query']
                
                # Tokenize query
                query_tensors = self.tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).input_ids.to(self.device)
                
                # Generate response
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs
                )
                
                # Decode response
                response_text = self.tokenizer.decode(
                    response_tensors[0],
                    skip_special_tokens=True
                )
                
                # Compute reward using toxicity model
                reward = self.compute_reward(response_text)
                
                # PPO step
                stats = ppo_trainer.step(
                    [query_tensors[0]],
                    [response_tensors[0]],
                    [reward]
                )
                
                # Log progress
                if i % 10 == 0:
                    print(f"Step {i}: Reward = {reward.item():.3f}")
                    print(f"  Query: {query[:50]}...")
                    print(f"  Response: {response_text[:50]}...")
                    print(f"  Stats: {stats}\n")
                
                # Save checkpoint
                if i > 0 and i % save_freq == 0:
                    checkpoint_dir = f"{self.output_dir}/checkpoint-{epoch}-{i}"
                    self.save_model(checkpoint_dir)
                    print(f"✓ Checkpoint saved to {checkpoint_dir}\n")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80 + "\n")
        
        # Save final model
        self.save_model(self.output_dir)
        print(f"✓ Final model saved to {self.output_dir}")
    
    def save_model(self, output_dir):
        """Save trained model and tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def evaluate_model(self, test_queries):
        """Evaluate model on test queries"""
        
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80 + "\n")
        
        total_reward = 0
        
        for i, query in enumerate(test_queries):
            # Tokenize
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7
                )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Get reward
            reward = self.compute_reward(response)
            total_reward += reward.item()
            
            print(f"\nTest {i+1}:")
            print(f"Query: {query}")
            print(f"Response: {response}")
            print(f"Reward: {reward.item():.3f}")
        
        avg_reward = total_reward / len(test_queries)
        print(f"\n{'='*80}")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"{'='*80}\n")
        
        return avg_reward


def main():
    """Main training pipeline"""
    
    # Initialize pipeline
    pipeline = PPOTrainingPipeline(
        model_name="gpt2",  # Start with smaller model for testing
        output_dir="./ppo_trained_chatbot"
    )
    
    # Prepare dataset from RLHF feedback
    print("Preparing training dataset...")
    dataset = pipeline.prepare_dataset(training_data_file="training_rewards.json")
    
    # Train
    pipeline.train(
        dataset=dataset,
        num_epochs=3,
        save_freq=50
    )
    
    # Evaluate
    test_queries = [
        "What services does Flowbotics offer?",
        "Tell me about pricing",
        "How can you help my business?"
    ]
    
    pipeline.evaluate_model(test_queries)
    
    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    main()