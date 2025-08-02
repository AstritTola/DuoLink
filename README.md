# DuoLink: A Unified Framework for Link Prediction via Line Graph Node Classification

## Overview

Link prediction is a foundational problem in network science with widespread applications ranging from social networks to biological systems. Despite the rise of powerful Graph Neural Networks (GNNs), existing models often fall short of classical heuristic methods on standard benchmarks. We identify two key challenges impeding progress: the prevalent two-stage approach—learning node embeddings followed by independent edge decoding—induces a representation misalignment with the prediction objective; moreover, current GNN architectures inadequately integrate valuable heuristic features, especially under feature heterophily.

DuoLink addresses these limitations by reformulating link prediction as a node classification task on the line graph of the original network. This novel perspective directly models edges as nodes, allowing end-to-end learning that naturally incorporates heuristic features and aligns with GNN inductive biases. Our framework leverages both graph neural networks and graph transformers to jointly capture structural heuristics and learned representations, enhancing expressivity and computational efficiency in recognizing key edge-level motifs.

## Features

- **End-to-end training** of link prediction using node classification on line graphs  
- Seamless **integration of classical heuristic features** with modern graph representation learning  
- Supports **heterophilic and homophilic** graph structures  
- Utilizes state-of-the-art **GNN and graph transformer architectures**  
- Theoretical guarantees on enhanced **expressivity and efficiency**  
- Empirical validation showing significant improvements over heuristic baselines and leading GNN models  

## Installation

```bash
git clone https://github.com/yourusername/duolink.git
cd duolink
pip install -r requirements.txt
