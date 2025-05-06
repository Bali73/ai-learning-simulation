# -*- coding: utf-8 -*-
"""
Created on Tue May  6 18:17:50 2025

@author: ADSU
"""

import numpy as np
import pandas as pd
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# -------------------------------
# Define Simulation Parameters
# -------------------------------
NUM_LEARNERS = 1000
SCENARIOS = ['Rural_Offline', 'Urban_Hybrid', 'Smart_Classroom']
LEARNING_MODES = ['Visual', 'Auditory', 'Text-Based']
BASE_KNOWLEDGE_RANGE = (20, 80)
LEARNING_RATE_RANGE = (0.01, 0.08)
COGNITIVE_FATIGUE_THRESHOLD = (3, 7)  # max lessons before fatigue
ENGAGEMENT_LEVEL_RANGE = (0.5, 1.0)

# -------------------------------
# Step 1: Generate Synthetic Learners
# -------------------------------
def generate_learner(id):
    scenario = np.random.choice(SCENARIOS, p=[0.4, 0.35, 0.25])
    base_knowledge = np.random.randint(*BASE_KNOWLEDGE_RANGE)
    learning_rate = np.round(np.random.uniform(*LEARNING_RATE_RANGE), 3)
    fatigue_limit = np.random.randint(*COGNITIVE_FATIGUE_THRESHOLD)
    engagement_level = np.round(np.random.uniform(*ENGAGEMENT_LEVEL_RANGE), 2)
    mode = random.choice(LEARNING_MODES)

    return {
        'LearnerID': id,
        'Scenario': scenario,
        'BaseKnowledge': base_knowledge,
        'LearningRate': learning_rate,
        'FatigueLimit': fatigue_limit,
        'EngagementLevel': engagement_level,
        'PreferredMode': mode
    }

learners = pd.DataFrame([generate_learner(i) for i in range(1, NUM_LEARNERS + 1)])

# -------------------------------
# Step 2: Simulate AI Interaction
# -------------------------------
def simulate_learning_session(row):
    sessions = 10
    knowledge = row['BaseKnowledge']
    history = []

    for s in range(sessions):
        fatigue_penalty = 0.9 if s >= row['FatigueLimit'] else 1.0
        scenario_penalty = {
            'Rural_Offline': 0.85,
            'Urban_Hybrid': 0.95,
            'Smart_Classroom': 1.0
        }[row['Scenario']]
        
        # AI adapts based on engagement + preferred mode match
        adaptation_score = (
            0.6 * row['EngagementLevel'] +
            0.4 * (1 if row['PreferredMode'] == 'Visual' else 0.9)
        )
        
        effective_learning = (
            knowledge + row['LearningRate'] * 100 * fatigue_penalty *
            scenario_penalty * adaptation_score
        )
        
        knowledge = min(100, round(effective_learning, 2))
        history.append(knowledge)

    return history

# Simulate sessions and attach results
learners['KnowledgeHistory'] = learners.apply(simulate_learning_session, axis=1)
learners['FinalKnowledge'] = learners['KnowledgeHistory'].apply(lambda x: x[-1])
learners['KnowledgeGain'] = learners['FinalKnowledge'] - learners['BaseKnowledge']

# -------------------------------
# Step 3: Analyze Summary Metrics
# -------------------------------
summary = learners.groupby('Scenario')['KnowledgeGain'].agg(['mean', 'std', 'count'])
print("\nKnowledge Gain Summary by Scenario:")
print(summary)

# Optional: Save output for model training or paper figures
learners.to_csv("simulated_learners_data.csv", index=False)

# Display first 5 entries
print("\nSample Learner Records:")
print(learners.head(5))
