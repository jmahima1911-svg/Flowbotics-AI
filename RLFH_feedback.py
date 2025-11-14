
"""
rlhf_ppo_trainer.py
Automated RLHF System with PPO Training using OpenRLHF
No manual human feedback - uses reward model for automated training
"""

import json
import os
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardModel:
    """Automated reward model for evaluating responses"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reward criteria weights
        self.weights = {
            'relevance': 0.30,
            'completeness': 0.25,
            'clarity': 0.20,
            'accuracy': 0.15,
            'engagement': 0.10
        }
        
        # Historical performance metrics
        self.performance_history = []
        
        logger.info(f"✓ Reward Model initialized on {self.device}")
    
    def evaluate_relevance(self, question: str, response: str, context: str) -> float:
        """Evaluate if response addresses the question"""
        question_lower = question.lower()
        response_lower = response.lower()
        
        # Extract key terms from question
        key_terms = set(question_lower.split())
        key_terms -= {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an'}
        
        # Check if response contains key terms
        matches = sum(1 for term in key_terms if term in response_lower)
        relevance_score = min(matches / max(len(key_terms), 1), 1.0)
        
        return relevance_score
    
    def evaluate_completeness(self, response: str, context: str) -> float:
        """Evaluate response completeness"""
        # Check for structured information
        has_structure = any(marker in response for marker in ['•', '*', '-', '1.', '2.'])
        
        # Check for pricing/details when context has them
        has_pricing = '$' in response if '$' in context else True
        has_details = len(response.split()) > 30
        
        completeness_score = (
            (0.4 if has_structure else 0.2) +
            (0.3 if has_pricing else 0.0) +
            (0.3 if has_details else 0.1)
        )
        
        return min(completeness_score, 1.0)
    
    def evaluate_clarity(self, response: str) -> float:
        """Evaluate response clarity"""
        # Check sentence structure
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Optimal sentence length: 15-25 words
        if 15 <= avg_sentence_length <= 25:
            length_score = 1.0
        elif avg_sentence_length < 10:
            length_score = 0.6  # Too short
        elif avg_sentence_length > 35:
            length_score = 0.5  # Too long
        else:
            length_score = 0.8
        
        # Check for formatting
        has_formatting = any(marker in response for marker in ['\n\n', '**', '##'])
        format_score = 1.0 if has_formatting else 0.7
        
        clarity_score = (length_score * 0.6 + format_score * 0.4)
        return clarity_score
    
    def evaluate_accuracy(self, response: str, context: str) -> float:
        """Evaluate factual accuracy based on context"""
        if not context:
            return 0.8  # Neutral score if no context
        
        # Check if response contains information from context
        context_terms = set(context.lower().split())
        response_terms = set(response.lower().split())
        
        # Calculate overlap
        overlap = len(context_terms & response_terms)
        accuracy_score = min(overlap / max(len(context_terms), 1) * 2, 1.0)
        
        # Penalize hallucinations (strong claims without context support)
        strong_claims = ['definitely', 'always', 'never', 'all', 'none', 'every']
        has_strong_claims = any(claim in response.lower() for claim in strong_claims)
        
        if has_strong_claims and accuracy_score < 0.5:
            accuracy_score *= 0.7  # Penalty for unsupported strong claims
        
        return accuracy_score
    
    def evaluate_engagement(self, response: str) -> float:
        """Evaluate response engagement"""
        # Check for conversational elements
        has_greeting = any(word in response.lower() for word in ['hey', 'hi', 'hello', 'great', 'perfect'])
        has_question = '?' in response
        has_empathy = any(word in response.lower() for word in ['understand', 'help', 'support', 'glad'])
        
        engagement_score = (
            (0.3 if has_greeting else 0.0) +
            (0.4 if has_question else 0.0) +
            (0.3 if has_empathy else 0.0)
        )
        
        return min(engagement_score, 1.0)
    
    def compute_reward(self, question: str, response: str, context: str = "") -> float:
        """Compute overall reward score"""
        
        scores = {
            'relevance': self.evaluate_relevance(question, response, context),
            'completeness': self.evaluate_completeness(response, context),
            'clarity': self.evaluate_clarity(response),
            'accuracy': self.evaluate_accuracy(response, context),
            'engagement': self.evaluate_engagement(response)
        }
        
        # Weighted average
        reward = sum(scores[k] * self.weights[k] for k in scores)
        
        # Normalize to [-1, 1] range
        reward = (reward * 2) - 1
        
        # Store performance
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'reward': reward,
            'scores': scores
        })
        
        return reward


class PPOTrainer:
    """PPO (Proximal Policy Optimization) Trainer for RLHF"""
    
    def __init__(
        self,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.reward_model = reward_model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon  # PPO clipping parameter
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Training history
        self.training_history = []
        self.episode_rewards = []
        
        logger.info("✓ PPO Trainer initialized")
    
    def compute_advantages(self, rewards: List[float], values: List[float]) -> np.ndarray:
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * 0.95 * gae  # 0.95 is lambda for GAE
            advantages.insert(0, gae)
        
        return np.array(advantages)
    
    def train_step(
        self,
        questions: List[str],
        responses: List[str],
        contexts: List[str],
        old_log_probs: List[float]
    ) -> Dict[str, float]:
        """Single PPO training step"""
        
        # Compute rewards
        rewards = [
            self.reward_model.compute_reward(q, r, c)
            for q, r, c in zip(questions, responses, contexts)
        ]
        
        # Compute value estimates (simplified)
        values = [np.mean(rewards)] * len(rewards)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO loss components (simplified for conceptual demonstration)
        policy_loss = -np.mean(advantages)  # Simplified
        value_loss = np.mean((np.array(rewards) - np.array(values)) ** 2)
        entropy_loss = -np.mean(np.log(np.abs(np.array(old_log_probs)) + 1e-8))
        
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        # Training metrics
        metrics = {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy_loss': float(entropy_loss),
            'total_loss': float(total_loss),
            'mean_reward': float(np.mean(rewards)),
            'mean_advantage': float(np.mean(advantages))
        }
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'num_samples': len(questions)
        })
        
        self.episode_rewards.extend(rewards)
        
        return metrics
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        if not self.episode_rewards:
            return {'status': 'No training data yet'}
        
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'recent_mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'total_training_steps': len(self.training_history)
        }


class AutomatedRLHFSystem:
    """Automated RLHF System - No manual feedback required"""
    
    def __init__(
        self,
        feedback_file: str = "rlhf_automated_data.json",
        model_checkpoint: str = "rlhf_model_checkpoint.pt"
    ):
        self.feedback_file = feedback_file
        self.model_checkpoint = model_checkpoint
        
        # Initialize components
        self.reward_model = RewardModel()
        self.ppo_trainer = PPOTrainer(self.reward_model)
        
        # Data storage
        self.training_data = self.load_training_data()
        self.batch_buffer = []
        
        logger.info("✓ Automated RLHF System initialized")
        logger.info(f"✓ Training samples loaded: {len(self.training_data)}")
    
    def load_training_data(self) -> List[Dict]:
        """Load training data"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_training_data(self):
        """Save training data"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2, ensure_ascii=False)
    
    def process_interaction(
        self,
        question: str,
        response: str,
        context: str = "",
        auto_train: bool = True
    ) -> Dict:
        """Process a single interaction and optionally train"""
        
        # Compute reward automatically
        reward = self.reward_model.compute_reward(question, response, context)
        
        # Create training sample
        sample = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response,
            'context': context,
            'reward': reward,
            'response_length': len(response),
            'response_format': self._detect_format(response)
        }
        
        # Add to buffer
        self.batch_buffer.append(sample)
        self.training_data.append(sample)
        
        # Train if buffer is full
        if auto_train and len(self.batch_buffer) >= 8:  # Batch size = 8
            self.train_on_batch()
        
        logger.info(f"Interaction processed - Reward: {reward:.3f}")
        
        return sample
    
    def train_on_batch(self):
        """Train on current batch"""
        if len(self.batch_buffer) < 2:
            return
        
        questions = [s['question'] for s in self.batch_buffer]
        responses = [s['response'] for s in self.batch_buffer]
        contexts = [s.get('context', '') for s in self.batch_buffer]
        old_log_probs = [np.random.normal(0, 1) for _ in self.batch_buffer]  # Placeholder
        
        # Train
        metrics = self.ppo_trainer.train_step(
            questions, responses, contexts, old_log_probs
        )
        
        logger.info(f"Batch training completed - Mean reward: {metrics['mean_reward']:.3f}")
        
        # Clear buffer
        self.batch_buffer = []
        
        # Save periodically
        if len(self.training_data) % 50 == 0:
            self.save_training_data()
    
    def _detect_format(self, response: str) -> str:
        """Detect response format"""
        if '•' in response or '*' in response or response.count('-') > 3:
            return 'bullet_points'
        elif '**' in response:
            return 'formatted'
        elif '\n\n' in response:
            return 'paragraphs'
        else:
            return 'simple'
    
    def get_statistics(self):
        """Display comprehensive statistics"""
        if not self.training_data:
            logger.info("No training data available yet.")
            return
        
        rewards = [s['reward'] for s in self.training_data]
        
        print("\n" + "="*80)
        print("AUTOMATED RLHF STATISTICS")
        print("="*80)
        print(f"\nTotal training samples: {len(self.training_data)}")
        print(f"Mean reward: {np.mean(rewards):.3f}")
        print(f"Std reward: {np.std(rewards):.3f}")
        print(f"Min reward: {np.min(rewards):.3f}")
        print(f"Max reward: {np.max(rewards):.3f}")
        
        # PPO Training stats
        print("\n" + "-"*80)
        print("PPO TRAINING STATISTICS")
        print("-"*80)
        stats = self.ppo_trainer.get_training_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Recent performance
        recent_rewards = rewards[-20:]
        print("\n" + "-"*80)
        print("RECENT PERFORMANCE (Last 20 samples)")
        print("-"*80)
        print(f"Mean: {np.mean(recent_rewards):.3f}")
        print(f"Trend: {'📈 Improving' if len(recent_rewards) > 10 and np.mean(recent_rewards[-10:]) > np.mean(recent_rewards[:10]) else '📉 Needs work'}")
        
        # Format preferences
        format_rewards = defaultdict(list)
        for sample in self.training_data:
            format_rewards[sample['response_format']].append(sample['reward'])
        
        print("\n" + "-"*80)
        print("FORMAT PERFORMANCE")
        print("-"*80)
        for fmt, rewards in format_rewards.items():
            print(f"  {fmt}: {np.mean(rewards):.3f} avg reward")
    
    def get_improvement_suggestions(self):
        """Get actionable improvement suggestions"""
        if len(self.training_data) < 10:
            print("Collecting more data for better suggestions...")
            return
        
        print("\n" + "="*80)
        print("IMPROVEMENT SUGGESTIONS")
        print("="*80)
        
        # Analyze recent vs old performance
        recent = self.training_data[-50:]
        old = self.training_data[:50] if len(self.training_data) > 100 else []
        
        if old:
            recent_avg = np.mean([s['reward'] for s in recent])
            old_avg = np.mean([s['reward'] for s in old])
            improvement = ((recent_avg - old_avg) / abs(old_avg)) * 100
            
            print(f"\n✓ Performance change: {improvement:+.1f}%")
            if improvement > 5:
                print("  Great! Model is learning effectively.")
            elif improvement < -5:
                print("  ⚠️ Performance declining. Consider adjusting hyperparameters.")
            else:
                print("  Stable performance. Continue training.")
        
        # Best practices from high-reward samples
        high_reward = [s for s in self.training_data if s['reward'] > 0.5]
        if high_reward:
            avg_length = np.mean([s['response_length'] for s in high_reward])
            common_format = max(
                set([s['response_format'] for s in high_reward]),
                key=[s['response_format'] for s in high_reward].count
            )
            
            print(f"\n✓ High-performing response patterns:")
            print(f"  - Optimal length: ~{avg_length:.0f} characters")
            print(f"  - Best format: {common_format}")


# Example usage
if __name__ == "__main__":
    # Initialize system
    rlhf_system = AutomatedRLHFSystem()
    
    # Simulate some interactions
    test_interactions = [
        {
            "question": "What are your pricing plans?",
            "response": "We offer three plans:\n• Starter: $99/month\n• Professional: $149/month\n• Enterprise: Custom pricing",
            "context": "Flowbotics offers Starter ($99), Professional ($149), and Enterprise plans."
        },
        {
            "question": "Do you support Slack integration?",
            "response": "Yes! We support Slack integration in all our plans.",
            "context": "Integrations: Google Sheets, Email, Slack, Microsoft Teams, Zapier"
        }
    ]
    
    # Process interactions
    for interaction in test_interactions:
        rlhf_system.process_interaction(**interaction)
    
    # Show statistics
    rlhf_system.get_statistics()
    rlhf_system.get_improvement_suggestions()