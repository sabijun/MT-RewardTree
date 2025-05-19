<p align='center'>
    <img src="images\title-MT-RewardTree.png">
</p>

<hr>

### Overview of MT-RewardTree

<p align='center'>
    <img src="images\MT-RewardTree_structure.png">
</p>

#### Feature

- We employ the Monte Carlo Tree Search (MCTS) method to generate token-level translation preference pairs (**Prefixed data**) for both model training and testing purposes. For contrast, we also utilize conventional approaches to generate sequence-level translation preference pairs (**Arbitrary data**). We partition both datasets into training and testing subsets for further evaluation.
- To assess the effectiveness of our custom dataset, we use prefixed data and  Arbitrary data to train our Implicit Process Reward Model. You can reach our models in <a href="https://huggingface.co/collections/sabijun/mt-rewardtree-models-67cac935143f75dfae6f0938" style="text-decoration: none; font-weight: bold;">Hugging Face</a>.
- Finally, we deploy our Implicit Process Reward Model in both Test-time Alignment and Hypothesis Ensembling frameworks, demonstrating significant performance improvements across evaluation metrics.
- We also provide the code for generating token-level prefixed translation preference pairs and utilizing our Process RM to obtain sequence-level results. You can access these codes in the code directory.
