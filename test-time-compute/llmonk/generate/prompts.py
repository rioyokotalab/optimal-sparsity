# https://medium.com/@ingridwickstevens/more-llm-acronyms-an-explainer-on-llama-3s-performance-benchmark-values-36722c6dcabb
# https://arxiv.org/abs/2206.14858
# https://github.com/EleutherAI/lm-evaluation-harness/blob/568af943e315100af3f00937bfd6947844769ab8/lm_eval/tasks/minerva_math/utils.py#L259
MATH_COT_PROMPT = """Problem:
Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."""

MATH_COT_2SHOT_PROMPT = """Problem:
Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct."""

Countdown_COT_PROMPT = """Question:
Using the numbers [19,36,55,7], create an equation that equals 65. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.

Answer:
<think>We need to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65, One possible combination is 55 + 36 - 19 + 7. Let's check: 55 + 36 = 91, 91 - 19 = 72, and 72 + 7 = 79. However, this doesn't equal 65. Another combination is 55 + 36 + 7 -19. Let's check: 55 + 36 = 91, 91 + 7 = 98, and 98 - 19 = 79. This also doesn't equal 65. After trying different combinations, I found that 55 + 36 + 7 - 19 = 79, which is close but not equal to 65. Finally, I found that 55 + 36 - 19 + 7 = 79, which is still not equal to 65. However, if we try 55 + 36 - 7 + 19, we get 55 + 36 = 91, 91 - 7 = 84, and 84 + 19 = 103, which is not equal to 65.After further trial and error, I found that 55 + 36 + 7 - 19 = 79, which is still not equal to 65. However, if we try 55 + 36 - 7 - 19, we get 55 + 36 = 91, 91 - 7 = 84, and 84 - 19 = 65. This equals 65. </think>
<answer>55 + 36 - 7 - 19</answer>
"""
GSM8K_COT_PROMPT = """
Question:
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

Answer:
There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question:
If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

Answer:
There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
"""
MATH_CODE_COT_PROMPT = """
def simple_math_problem():
    '''
    A parabola with equation y = x^2 + b*x + c passes through the points
    (-1, -11) and (3, 17). What is c?
    '''
    # Plug in (-1, -11):  (-1)^2 + b*(-1) + c = -11  →  1 - b + c = -11  →  -b + c = -12
    # Plug in (3, 17):   3^2 + 3b + c = 17         →  9 + 3b + c = 17   →   3b + c = 8
    # We have two equations:
    #   (1) -b + c = -12
    #   (2)  3b + c =   8
    # Subtract (1) from (2): 3b + c - ( -b + c ) = 8 - ( -12 )  →  4b = 20  →  b = 5
    b = 20 / 4  # b = 5
    # Then from (1): -b + c = -12  →  c = b - 12
    c = b - 12  # c = 5 - 12 = -7
    result = c  # result == -7
    return result

def simple_math_problem():
    '''
    At the 2007 Math Olympics, Team Canada won 17 out of a possible 100 medals.
    Which one of the following is closest to the fraction of medals that they won?
    1/4, 1/5, 1/6, 1/7, 1/8
    '''
    from sympy import Rational, Abs
    import numpy as np

    # fraction of medals Canada won
    medal_frac = Rational(17, 100)

    # the five choices
    options = [
        Rational(1, 4),
        Rational(1, 5),
        Rational(1, 6),
        Rational(1, 7),
        Rational(1, 8),
    ]

    # compute absolute differences
    diffs = [Abs(medal_frac - opt) for opt in options]  # [2/25, 3/100, 1/300, 19/700, 9/200]

    # find the index of the smallest difference
    min_idx = int(np.argmin([float(d) for d in diffs]))

    # select the closest fraction
    result = options[min_idx]   # result == 1/6

    return result

def simple_math_problem():
    '''
    A rectangular box P is inscribed in a sphere of radius r. The surface area of P is 384,
    and the sum of the lengths of its 12 edges is 112. What is r?
    '''
    # We have:
    #   2(xy + xz + yz) = 384    (surface area)
    #   4(x + y + z) = 112       (sum of edges)
    #   x^2 + y^2 + z^2 = (2r)^2 (box diagonal = sphere diameter)
    from sympy import symbols, Eq, solve
    x, y, z, r = symbols('x y z r')
    eq1 = Eq(2 * (x*y + x*z + y*z), 384)
    eq2 = Eq(4 * (x + y + z), 112)
    eq3 = Eq(x**2 + y**2 + z**2, (2*r)**2)
    sols = solve((eq1, eq2, eq3), (x, y, z, r))
    # select the positive radius
    for sol in sols:
        if sol[r] > 0:
            result = sol[r]
            break
    return result

def simple_math_problem():
    '''
    A bee starts flying from point P0. She flies 1 inch due east to point P1.
    For j ≥ 1, once the bee reaches point Pj, she turns 30° counterclockwise
    and then flies j+1 inches straight to point P_{j+1}. When the bee reaches
    P_{2015}, how far from P0 is she, in inches?
    '''
    from sympy import exp, I, pi, sqrt

    # rotation by 30° each step
    omega = exp(I * pi/6)

    # accumulate complex displacement
    position = 0
    for j in range(2015):
        step = (j + 1) * omega**j
        position += step

    # distance from origin
    real, imag = position.as_real_imag()
    dist = sqrt(real**2 + imag**2)

    # simplify expression
    result = dist  # result == sqrt(2)*(1008 + 1008*sqrt(3))
    return result
"""

GSM8K_CODE_COT_PROMPT = """
def simple_math_problem():
    '''
    Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    '''

    eggs_per_day = 16
    eggs_for_breakfast = 3
    eggs_baked_for_friends = 4
    eggs_sold = (eggs_per_day - eggs_for_breakfast - eggs_baked_for_friends) # eggs_sold == 7
    price_per_egg = 2
    total_revenue = eggs_sold * price_per_egg # total_revenue == 14
    result = total_revenue # result == 14
    return result

def simple_math_problem():
    '''
    During the hot dog eating contest, the 1st competitor ate 12 hot dogs in 3 minutes.  The 2nd competitor ate twice that amount and the 3rd competitor ate 25% less than the 2nd competitor.  How many hotdogs did the 3rd competitor eat?
    '''
    
    hot_dogs_1st_competitor = 12
    hot_dogs_2nd_competitor = 2 * hot_dogs_1st_competitor # hot_dogs_2nd_competitor == 24
    hot_dogs_3rd_competitor = hot_dogs_2nd_competitor * 0.75 # hot_dogs_3rd_competitor == 18.0
    result = hot_dogs_3rd_competitor # result == 18.0
    return result

"""

# Examples taken from the MiniF2F training set
# https://github.com/rah4927/lean-dojo-mew/blob/main/MiniF2F/Validation.lean
MINIF2F_FEW_SHOT_PROMPT = """```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem1
  (k x: ℝ)
  (h₀ : x = (13 - Real.sqrt 131) / 4)
  (h₁ : 2 * x^2 - 13 * x + k = 0) :
  k = 19/4 :=
by (
  rw [h₀] at h₁
  rw [eq_comm.mp (add_eq_zero_iff_neg_eq.mp h₁)]
  norm_num
  rw [pow_two]
  rw [mul_sub]
  rw [sub_mul, sub_mul]
  rw [Real.mul_self_sqrt _]
  ring
  linarith
)
```
Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.
```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem2
  (x p : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p) :
  x - p = 2 - 2 * p :=
by (
  suffices abs (x - 2) = -(x - 2) by
    rw [h₁] at this
    linarith
  apply abs_of_neg
  linarith
)
```
Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.
```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem3
  (x : ℝ)
  (f g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x + 2)
  (h₁ : ∀ x, g x = x^2)
  (h₂ : f (g x) = g (f x)) :
  x = - 1/2 :=
by (
  norm_num
  simp_all [-one_div]
  field_simp [h₁]
  linarith
)
```
Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.
```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem4
  (a b : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a ≠ 2 * b)
  (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5) :
  (a + 11 * b) / (a - b) = 2 :=
by (
  rw [eq_comm]
  refine' (eq_div_iff _).mpr _
  exact sub_ne_zero_of_ne h₀
  rw [eq_comm] at h₂
  suffices : a = 13 * b; linarith
  have key : 5 * (a - 2 * b) = 4 * a + 3 * b; rwa [(eq_div_iff (sub_ne_zero_of_ne h₁)).mp]
  linarith
)
```
Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.
```import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem5
  Int.floor ((9:ℝ) / 160 * 100) = 5 :=
by (
  rw [Int.floor_eq_iff]
  constructor
  all_goals norm_num
)
```

"""
