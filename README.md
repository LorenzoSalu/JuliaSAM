# JuliaSAM
JuliaSAM is a native Julia implementation of Meta’s Segment Anything Model (SAM), designed to provide high-quality, flexible image segmentation within Julia-based pipelines.

While Meta’s original SAM is only available in Python, this translation to Julia removes cross-language overhead, offering improved performance and tighter integration with Julia workflows.

This project is still under development.

# 


## Installation & Setup
- Step 1 - Install `JuliaSAM.jl`:
```bash
(@v1.8) pkg > add https://github.com/LorenzoSalu/JuliaSAM.git
```
- Step 2 - install dependencies:
```bash
julia setup.jl
```

## Prospective developments
The following improvements and extensions are planned for future versions of JuliaSAM:
	•	Codebase completion
Finalize and integrate missing components to fully align with Meta’s original SAM specification.
	•	Robustness improvements
Refine early-stage modules to ensure consistent and stable behavior across a wide range of inputs.
	•	Performance optimization
	•	Reduce memory usage and execution time
	•	Streamline computational flows
	•	Improve internal module and dependency organization
	•	LoRA integration
Implement Low-Rank Adaptation (LoRA) to enable efficient fine-tuning of SAM for domain-specific tasks such as nucleus segmentation.
	•	Automatic prompt generation
Develop a prompt generation module to automatically guide the model in detecting cell nuclei.
	•	Segment Any Cell compatibility
Incorporate methods inspired by the Segment Any Cell project to enhance segmentation accuracy in biomedical applications.
	•	SOPHYSM integration
Integrate JuliaSAM into the SOPHYSM platform to support usage by non-expert users and enable advanced cancer spatial analysis workflows in the J-Space pipeline.
