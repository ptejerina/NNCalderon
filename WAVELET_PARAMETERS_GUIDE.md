# Wavelet Boundary Conditions Parameter Guide

## Quick Answer: How to Play with Parameters

You have **3 main knobs** to create variety in 50-60 wavelet boundary conditions:

### 1. **Width Parameter `a`** (Narrowness of wavelets)
```python
# Narrow, sharp wavelets (localized energy)
a = 0.8   # Very narrow
a = 1.2   # Moderately narrow

# Wide, smooth wavelets (spread-out energy)
a = 1.8   # Moderately wide
a = 2.5   # Very wide
```

**Math**: The Ricker wavelet is defined as:
$$f(x) = (1-2\pi^2 a^2 (x-x_0)^2) \cdot e^{-\pi^2 a^2 (x-x_0)^2}$$

- **Smaller `a`** → narrower peak, high sharpness
- **Larger `a`** → wider peak, more diffuse

### 2. **Amplitude** (Strength of the signal)
```python
# Weak signals
amplitude = 0.5   # Subtle boundary effects
amplitude = 0.8   # Mild effects

# Strong signals
amplitude = 1.2   # Pronounced effects
amplitude = 1.5   # Very strong
```

- **Lower amplitude** → smaller potential variations
- **Higher amplitude** → larger potential variations

### 3. **Side Distribution** (Which boundaries get wavelets)
```python
# Single-side configurations (4 options)
sides = ['bottom']    # Only bottom has wavelet, others are 0
sides = ['right']     # Only right
sides = ['top']       # Only top
sides = ['left']      # Only left

# Pair configurations (6 options)
sides = ['bottom', 'right']
sides = ['bottom', 'top']
sides = ['left', 'right']
# ... (6 total pairs)

# Multi-side configurations
sides = ['bottom', 'right', 'top']  # Three sides
sides = ['bottom', 'right', 'top', 'left']  # All sides

# Multiple wavelets on same side
# Instead of 1 wavelet at x0=0.5, use 2 wavelets at x0=[1/3, 2/3]
```

### 4. **Center Position** (Where along each side)
```python
# Single centered
x0 = 0.5

# Three positions (left-center-right)
x0 = [1/3, 0.5, 2/3]

# Custom positions
x0 = [0.25, 0.5, 0.75]
```

---

## Quick Example: Generate 60 Diverse BCs

### Step 1: Define Parameter Ranges
```python
width_values = [0.8, 1.2, 1.8]        # 3 choices
amplitude_values = [0.7, 1.0, 1.3]    # 3 choices
position_values = [0.5]                # 1 choice
```

### Step 2: Create Different Configuration Types

| Type | Description | Count |
|------|-------------|-------|
| Single side | 1 wavelet on each side | 4 × 3 × 3 = 36 |
| Pair sides | Wavelets on 2 sides | 6 × 2 × 2 = 24 |
| Dual same-side | 2 wavelets on one side | 4 × 2 = 8 |
| **Total** | Combined | **~60+** |

---

## How Each Parameter Affects Results

### Effect of `a` (Width)
- `a = 0.8`: Sharp, localized peak → good for capturing point-like features
- `a = 1.5`: Moderate spread → balanced representation
- `a = 2.5`: Broad, smooth → captures large-scale features

**Use case**: Mix all three to get wavelets from "pinpoint" to "smooth"

### Effect of Amplitude
- `amp = 0.7`: Subtle → good for noise-like perturbations
- `amp = 1.0`: Standard → good baseline
- `amp = 1.5`: Strong → good for testing large boundary effects

**Use case**: Include all for robustness across scales

### Effect of Sides
- Single side: Most localized, 4 natural options
- Pairs: More complex interactions
- All sides: Maximum complexity, 1 option

**Use case**: Single + Pairs + 3-sides covers most diversity

### Effect of Multiple Wavelets on Same Side
- 1 wavelet: Simple, centered
- 2 wavelets: Creates interference patterns (more interesting!)
- 3 wavelets: Very complex boundary patterns

**Use case**: Add a few to get non-trivial interactions

---

## Practical Strategy: From 0-60 BCs

### Phase 1: Foundation (20 BCs)
- 4 sides × 1 width × 1 amplitude × 1 position (single per side) = **4 BCs**
- 6 side-pairs × 2 widths × 2 amplitudes = **24 BCs**
- **Total: 28 BCs** (take best 20)

### Phase 2: Add Width Variety (15 BCs)
- 4 sides × 3 widths × 1 amplitude = **12 BCs** (add 12 more)
- **Total: 40 BCs**

### Phase 3: Add Amplitude Variety (10 BCs)
- Single side with mixed amplitudes: 4 × 3 = **12 BCs** (add 10 more)
- **Total: 50 BCs**

### Phase 4: Add Complex Patterns (10 BCs)
- Dual wavelets per side: 4 × 2 = **8 BCs** (add 8 more)
- 3-side wavelets: 4 × 1 = **4 BCs** (add 4 more)
- **Total: 60+ BCs** ✓

---

## Code Template

```python
from itertools import combinations
import numpy as np

# Define ranges
a_values = [0.8, 1.2, 1.8, 2.3]
amp_values = [0.6, 0.85, 1.0, 1.25, 1.5]
x0_values = [1/3, 0.5, 2/3]
sides_list = ['bottom', 'right', 'top', 'left']

configs = []

# Type 1: Single wavelet per side
for side in sides_list:
    for a in a_values[:2]:  # Use first 2 a values
        for amp in amp_values:
            for x0 in x0_values:
                configs.append({
                    'type': 'single',
                    'sides': [side],
                    'a': a, 'amplitude': amp, 'x0': [x0]
                })

# Type 2: Pairs of sides
for s1, s2 in combinations(sides_list, 2):
    for a in a_values[1:3]:  # Use middle a values
        for amp in amp_values[::2]:  # Every other amplitude
            configs.append({
                'type': 'pair',
                'sides': [s1, s2],
                'a': a, 'amplitude': amp, 'x0': [0.5, 0.5]
            })

# Type 3: Dual wavelets on same side
for side in sides_list:
    for a in a_values:
        for amp in amp_values[::2]:
            configs.append({
                'type': 'dual_same_side',
                'sides': [side],
                'a': a, 'amplitude': amp, 'x0': [1/3, 2/3]
            })

print(f"Total configurations: {len(configs)}")
```

---

## FAQ

### Q: Which parameter has the biggest effect?
**A**: **Width `a`** - it changes the wavelet shape dramatically. Varying `a` is your primary knob.

### Q: Should I vary all parameters equally?
**A**: No! Focus on:
1. **Width `a`**: Vary heavily (3-4 values)
2. **Amplitude**: Vary moderately (2-3 values)
3. **Sides**: Use all combinations (4 + 6 pairs + 4 triples = 14)
4. **Position**: Use 1-2 values (center + maybe one offset)

### Q: How do I know I have enough variety?
**A**: Plot the first 20 boundary conditions. If they look visibly different from each other, you have good variety.

### Q: Can I add randomness?
**A**: Yes! After generating 50-60 deterministic configs, you can add:
```python
# Random width in range
a_random = np.random.uniform(0.8, 2.5)

# Random amplitude in range
amp_random = np.random.uniform(0.6, 1.5)

# Random centers
x0_random = np.random.uniform(0.2, 0.8, size=2)
```

---

## Integration with Your Code

**Key Point**: You DON'T need to modify `generate_dtn_data`. Instead:

1. Generate configs using the parameter space (this notebook)
2. Convert configs to boundary arrays
3. Call `solver.solve()` directly for each BC
4. Save results manually to NPZ

This gives you complete control without touching the original function.

---

## See Also
- `Wavelet_BC_Parameter_Guide.ipynb` - Full implementation
- `playground_inverted gaussian wavelets.ipynb` - Original code reference
