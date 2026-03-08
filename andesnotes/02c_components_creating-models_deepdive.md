# ANDES Study Notes — Components + Creating Models Deep Dive

Sources:
- components: parameters, services, discrete, blocks, groups
- creating-models: model structure, static example, testing

## Parameters
- `NumParam`: scalar/numeric model parameters
- `IdxParam`: index/link parameters for cross-model references
- External parameter access through index/group mechanisms

## Services and Logic
- Services provide computed intermediate values
- Discrete logic controls event-style behavior and switching

## Blocks and Groups
- Blocks encapsulate reusable control patterns
- Groups organize model families and shared interfaces

## Creating a New Model
Recommended sequence:
1. Define structure and parameters
2. Declare variables and equations
3. Register model in framework
4. Run preparation / code-generation steps
5. Validate with minimal test cases

## Testing Priority
Prefer a minimal reproducible model first, then expand complexity.
