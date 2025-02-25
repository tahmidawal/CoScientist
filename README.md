# CoScientist: AI Co-Scientist for Autonomous Research

CoScientist is an advanced AI system capable of autonomously generating novel and high-impact research ideas in Machine Learning and related fields. It follows a similar architecture and methodology to Google's AI Co-Scientist.

## Green's Function Research Ideas

This repository includes three novel research ideas for learning Green's functions without resolution limitations using deep learning:

1. **Uncertainty Quantification (UQ) for Green's Functions**: Integrating UQ methods with deep learning models to provide confidence estimates for Green's function predictions at arbitrary resolutions.

2. **Meta-Learning for Green's Functions**: Using meta-learning techniques to enable models that quickly adapt to learn Green's functions for new PDEs or domain geometries with minimal data.

3. **Graph Neural Networks for Complex Domains**: Utilizing GNNs to learn Green's functions on irregular or complex domains by representing the domain as a graph.

Each research idea comes with a detailed implementation plan, including technical approach, required datasets, algorithms with pseudocode, evaluation methodologies, and academic references.

## Features

- **Autonomous Idea Generation**: Generate innovative research hypotheses based on a research goal
- **Iterative Refinement**: Refine hypotheses through multiple stages of evaluation and improvement
- **Evolutionary Optimization**: Evolve hypotheses using evolutionary algorithms to increase quality
- **Multi-Agent Architecture**: Specialized agents for different aspects of the research process
- **Flexible Workflows**: Standard, custom, and iterative workflows to suit different research needs
- **Multiple LLM Support**: Integration with OpenAI, Anthropic, Hugging Face, Google Gemini, and PraisonAI models

## Installation

### Basic Installation

```bash
pip install coscientist
```

### Installation with Additional Dependencies

For specific LLM integrations:

```bash
# Install with OpenAI support
pip install "coscientist[openai]"

# Install with Anthropic (Claude) support
pip install "coscientist[anthropic]"

# Install with Google Gemini support
pip install "coscientist[gemini]"

# Install with all integrations
pip install "coscientist[all]"
```

### Development Installation

```bash
git clone https://github.com/yourusername/coscientist.git
cd coscientist
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

```bash
# Run with the standard workflow
coscientist --research-goal "Develop a neural network architecture for solving differential equations" --output results.json

# Generate research ideas for Green's functions
python examples/top_ideas_with_research.py --research-goal "Using deep learning to learn Green's functions without resolution limitations" --output-dir greens_function_research
```

### Python API

```python
from coscientist.core.coscientist import CoScientist

# Create a CoScientist instance
co_scientist = CoScientist(research_goal="Develop a new generative model for image synthesis")

# Run the standard workflow
result = co_scientist.run_full_workflow(num_hypotheses=5, generations=3)

# Print the summary
print(result["summary"])
```

## System Architecture

CoScientist is built on a multi-agent architecture with specialized components:

1. **Hypothesis Generation Agent**: Creates novel research hypotheses based on the research goal
2. **Reflection Agent**: Evaluates and refines hypotheses for theoretical soundness and feasibility
3. **Evolution Agent**: Uses evolutionary algorithms to optimize and combine hypotheses
4. **Ranking Agent**: Evaluates and ranks hypotheses based on multiple criteria
5. **Meta-Review Agent**: Generates a comprehensive research overview from the top-ranked hypotheses

## Workflows

CoScientist supports different workflow types:

- **Standard Workflow**: Generate → Refine → Evolve → Rank → Meta-Review
- **Custom Workflow**: Define your own sequence of steps
- **Iterative Workflow**: Run multiple iterations until convergence

## Configuration

You can configure CoScientist using JSON configuration files:

```json
{
  "agents": {
    "generation": {
      "model": "openai:gpt-4",
      "params": {
        "temperature": 0.8
      }
    },
    "reflection": {
      "model": "anthropic:claude-3-opus-20240229",
      "params": {
        "temperature": 0.5
      }
    }
  }
}
```

## Examples

See the `examples/` directory for example scripts demonstrating different use cases:

- `examples/generate_research_ideas.py`: Basic usage examples
- `examples/custom_workflow.py`: Creating custom workflows
- `examples/iterative_optimization.py`: Running iterative optimization workflows

## Inspiration

This project is inspired by:

- [Google's AI Co-Scientist Blog](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)
- [Google's AI Co-Scientist Paper](https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf)
- [PraisonAI Documentation](https://docs.praison.ai/tools/tools)

## License

MIT License 