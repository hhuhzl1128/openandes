# ANDES Study Notes — Modeling Concepts & Variables Deep Dive

Sources:
- framework-overview
- dae-formulation
- variables

## Hybrid Symbolic–Numeric Design
Main idea:
- Express equations symbolically first
- Let the framework generate Jacobians and optimized numerical routines

Benefits:
- Faster prototyping
- Lower manual derivative burden
- Better consistency between model expression and solver inputs

## DAE Perspective
ANDES models are represented through differential and algebraic equations, with explicit variable typing and update paths.

## Variable Layering
- Internal state/algebraic variables
- External references and services
- Structured access for model coupling

## Practical Guidance
When debugging, verify variable scope, dependency direction, and initialization order before changing solver settings.
