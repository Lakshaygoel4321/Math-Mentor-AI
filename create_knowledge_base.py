import os

# Create directory
os.makedirs("rag/knowledge_base", exist_ok=True)

# Algebra formulas
algebra_content = """Quadratic Formula: For ax² + bx + c = 0, x = (-b ± √(b²-4ac)) / 2a
Discriminant: b² - 4ac determines the nature of roots
If b² - 4ac > 0: Two distinct real roots
If b² - 4ac = 0: One repeated real root
If b² - 4ac < 0: Two complex conjugate roots

Factorization: x² + (a+b)x + ab = (x+a)(x+b)
Difference of squares: a² - b² = (a+b)(a-b)
Perfect square: a² + 2ab + b² = (a+b)²

Binomial Theorem: (a+b)ⁿ = Σ(nCr × aⁿ⁻ʳ × bʳ) where r goes from 0 to n
Exponent rules: aᵐ × aⁿ = aᵐ⁺ⁿ, aᵐ / aⁿ = aᵐ⁻ⁿ, (aᵐ)ⁿ = aᵐⁿ
"""

# Calculus formulas
calculus_content = """Derivative Rules:
- Power Rule: d/dx(xⁿ) = n×xⁿ⁻¹
- Constant Rule: d/dx(c) = 0
- Sum Rule: d/dx(f+g) = f' + g'
- Product Rule: d/dx(uv) = u(dv/dx) + v(du/dx)
- Quotient Rule: d/dx(u/v) = (v(du/dx) - u(dv/dx))/v²
- Chain Rule: d/dx(f(g(x))) = f'(g(x))×g'(x)

Common Derivatives:
- d/dx(sin x) = cos x
- d/dx(cos x) = -sin x
- d/dx(eˣ) = eˣ
- d/dx(ln x) = 1/x

Limits:
- lim(x→0) sin(x)/x = 1
- lim(x→0) (1-cos(x))/x = 0
- lim(x→∞) (1 + 1/x)ˣ = e
- lim(x→0) (eˣ-1)/x = 1

Integration Rules:
- ∫xⁿ dx = xⁿ⁺¹/(n+1) + C, where n ≠ -1
- ∫1/x dx = ln|x| + C
- ∫eˣ dx = eˣ + C
- ∫sin x dx = -cos x + C
- ∫cos x dx = sin x + C
"""

# Probability formulas
probability_content = """Probability Basics:
- P(A∪B) = P(A) + P(B) - P(A∩B) (Addition rule)
- P(A∩B) = P(A)×P(B) if A and B are independent
- P(A|B) = P(A∩B) / P(B) (Conditional probability)
- P(Aᶜ) = 1 - P(A) (Complement rule)

Bayes Theorem: P(A|B) = P(B|A)×P(A) / P(B)

Permutations: nPr = n!/(n-r)! 
- Number of ways to arrange r items from n items (order matters)

Combinations: nCr = n!/(r!(n-r)!)
- Number of ways to choose r items from n items (order doesn't matter)

Expected Value: E(X) = Σ(x×P(x)) for discrete random variables
Variance: Var(X) = E(X²) - [E(X)]²
Standard Deviation: σ = √Var(X)

Binomial Distribution: P(X=k) = nCk × pᵏ × (1-p)ⁿ⁻ᵏ
- n trials, probability p of success, k successes
"""

# Linear algebra basics
linear_algebra_content = """Matrix Operations:
- Matrix addition: Add corresponding elements
- Matrix multiplication: (AB)ᵢⱼ = Σ(Aᵢₖ × Bₖⱼ)
- Transpose: (Aᵀ)ᵢⱼ = Aⱼᵢ
- Identity matrix: I × A = A × I = A

Determinant (2×2): |A| = ad - bc for A = [[a,b],[c,d]]
Determinant (3×3): Use cofactor expansion

Matrix Inverse: A⁻¹ exists if det(A) ≠ 0
- A × A⁻¹ = A⁻¹ × A = I
- For 2×2: A⁻¹ = (1/det(A)) × [[d,-b],[-c,a]]

System of Linear Equations:
- AX = B can be solved as X = A⁻¹B (if A⁻¹ exists)
- Cramer's rule: xᵢ = det(Aᵢ)/det(A)

Eigenvalues and Eigenvectors:
- AV = λV where λ is eigenvalue, V is eigenvector
- Find λ by solving det(A - λI) = 0
"""

# Write files with UTF-8 encoding
files = {
    "algebra_formulas.txt": algebra_content,
    "calculus_formulas.txt": calculus_content,
    "probability_formulas.txt": probability_content,
    "linear_algebra_formulas.txt": linear_algebra_content
}

for filename, content in files.items():
    filepath = os.path.join("rag/knowledge_base", filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"✅ Created: {filepath}")

print("\n✅ All knowledge base files created successfully!")
