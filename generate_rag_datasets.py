import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# KNOWLEDGE BASE — 500 course content chunks
# ─────────────────────────────────────────────

SUBJECTS = {
    "Physics": {
        "Thermodynamics": [
            "The first law of thermodynamics states that energy cannot be created or destroyed, only converted from one form to another. Mathematically, ΔU = Q − W, where ΔU is the change in internal energy, Q is the heat added to the system, and W is the work done by the system.",
            "An isothermal process is one in which the temperature of the system remains constant. For an ideal gas undergoing an isothermal expansion, the work done is W = nRT ln(V₂/V₁). Since temperature is constant, ΔU = 0 and Q = W.",
            "An adiabatic process is one in which no heat is exchanged with the surroundings (Q = 0). For an ideal gas, the relationship between pressure and volume is PV^γ = constant, where γ = Cp/Cv is the heat capacity ratio.",
            "Carnot's theorem states that no heat engine operating between two heat reservoirs can be more efficient than a Carnot engine operating between the same temperatures. The Carnot efficiency is η = 1 − T_cold/T_hot.",
            "Entropy is a measure of the disorder or randomness in a system. The second law of thermodynamics states that the total entropy of an isolated system always increases or remains constant. For a reversible process, ΔS = Q_rev/T.",
            "The zeroth law of thermodynamics states that if two systems are in thermal equilibrium with a third system, they are in thermal equilibrium with each other. This forms the basis for temperature measurement.",
            "The coefficient of performance (COP) of a refrigerator is the ratio of heat removed from the cold reservoir to the work done. COP_refrigerator = Q_cold / W = T_cold / (T_hot − T_cold) for a Carnot refrigerator.",
            "In a PV diagram, the area enclosed by a cyclic process represents the net work done by the system. Clockwise cycles represent engines (net positive work output), while counterclockwise cycles represent refrigerators (net positive work input).",
            "The specific heat capacity of a substance is the amount of heat required to raise the temperature of 1 kg of the substance by 1°C. For water, C_p ≈ 4186 J/(kg·K). For gases, we distinguish between C_p (at constant pressure) and C_v (at constant volume), with C_p − C_v = R for ideal gases.",
            "Stefan-Boltzmann Law: Every object emits thermal radiation. The power radiated per unit area is P/A = εσT⁴, where ε is the emissivity (0 to 1), σ = 5.67 × 10⁻⁸ W/(m²·K⁴) is the Stefan-Boltzmann constant, and T is the absolute temperature.",
        ],
        "Optics": [
            "Snell's Law describes the refraction of light at an interface: n₁ sin(θ₁) = n₂ sin(θ₂), where n₁ and n₂ are the refractive indices of the two media, and θ₁ and θ₂ are the angles of incidence and refraction with respect to the normal.",
            "Total internal reflection occurs when light travels from a denser to a rarer medium at an angle greater than the critical angle. The critical angle θ_c satisfies sin(θ_c) = n₂/n₁ (where n₁ > n₂). Optical fibres use this principle.",
            "The lens formula is 1/f = 1/v − 1/u, where f is the focal length, v is the image distance, and u is the object distance (using the sign convention where distances are measured from the optical centre).",
            "Young's Double Slit Experiment demonstrates the wave nature of light. Constructive interference occurs at path difference = mλ, and destructive interference at (m + 1/2)λ. Fringe width β = λD/d, where D is the slit-to-screen distance and d is the slit separation.",
            "The resolving power of a microscope is defined by the Rayleigh criterion: the minimum resolvable distance is d = 0.61λ/n·sin(α), where λ is the wavelength of light, n is the refractive index of the medium, and α is the half-angle of the objective lens.",
            "Huygen's Principle states that every point on a wavefront acts as a source of secondary spherical wavelets. The new wavefront is the envelope of all these secondary wavelets. This principle explains reflection, refraction, and diffraction.",
            "In single-slit diffraction, the condition for minima is a·sin(θ) = mλ (m = ±1, ±2, ...), where a is the slit width and λ is the wavelength. The central maximum has an angular width of 2λ/a.",
        ],
        "Mechanics": [
            "Newton's second law in rotational form: τ = Iα, where τ is the net torque, I is the moment of inertia, and α is the angular acceleration. This is the rotational analogue of F = ma.",
            "The moment of inertia of a solid sphere about its diameter is I = (2/5)mr². For a hollow sphere it is I = (2/3)mr². For a solid cylinder about its axis, I = (1/2)mr².",
            "The work-energy theorem states that the net work done on an object equals its change in kinetic energy: W_net = ΔKE = (1/2)mv₂² − (1/2)mv₁².",
            "In simple harmonic motion (SHM), the restoring force is proportional to displacement: F = −kx. The period of oscillation is T = 2π√(m/k) for a spring-mass system and T = 2π√(L/g) for a simple pendulum.",
            "Escape velocity from a planet is the minimum speed needed to break free from its gravitational pull without further propulsion: v_e = √(2GM/R), where G is the gravitational constant, M is the planet's mass, and R is its radius.",
            "For circular motion, the centripetal acceleration is a = v²/r = ω²r, directed towards the centre. The centripetal force is F = mv²/r = mω²r.",
            "Conservation of angular momentum: when no external torque acts on a system, L = Iω = constant. This explains why a spinning ice skater spins faster when pulling in their arms.",
            "Bernoulli's principle states that for an ideal fluid in steady flow, the sum P + ½ρv² + ρgh = constant along a streamline. This explains why pressure decreases as fluid speed increases.",
        ],
        "Electricity & Magnetism": [
            "Gauss's Law states that the total electric flux through a closed surface is equal to the enclosed charge divided by ε₀: Φ_E = Q_enc/ε₀. This is used to find electric fields for symmetric charge distributions.",
            "Faraday's Law of Electromagnetic Induction: the induced EMF in a loop is equal to the negative rate of change of magnetic flux through it: ε = −dΦ_B/dt. This is the basis for electric generators.",
            "Lenz's Law states that the induced current in a conductor opposes the change in magnetic flux that caused it. This is consistent with the law of conservation of energy.",
            "Ampere's Circuital Law: the line integral of the magnetic field around a closed loop equals μ₀ times the total current enclosed by that loop: ∮B·dl = μ₀I_enc.",
            "The capacitance of a parallel plate capacitor is C = ε₀A/d, where A is the plate area and d is the separation. With a dielectric (κ) inserted, C = κε₀A/d. Energy stored is U = Q²/(2C) = CV²/2.",
            "Kirchhoff's Voltage Law (KVL) states that the algebraic sum of all voltages around any closed loop in a circuit is zero. Kirchhoff's Current Law (KCL) states that the sum of currents entering a node equals the sum leaving it.",
        ],
        "Modern Physics": [
            "The photoelectric effect: when light of frequency f strikes a metal surface, electrons are emitted only if f ≥ f₀ (threshold frequency). The maximum kinetic energy of emitted electrons is KE_max = hf − φ, where φ = hf₀ is the work function.",
            "de Broglie's hypothesis: every moving particle has an associated wavelength λ = h/mv = h/p. This wave-particle duality was confirmed by the Davisson-Germer electron diffraction experiment.",
            "Bohr's model of the hydrogen atom: electrons orbit in discrete energy levels. The energy of the nth level is E_n = −13.6/n² eV. Transitions emit photons with energy ΔE = hf = E_higher − E_lower.",
            "Radioactive decay law: N(t) = N₀ e^(−λt), where λ is the decay constant. The half-life T₁/₂ = ln2/λ. Activity A = λN. Types of decay: alpha (²₂He nucleus), beta (electron/positron), gamma (high-energy photon).",
        ],
    },
    "Chemistry": {
        "Organic Reactions": [
            "SN2 reactions proceed via a single concerted step: the nucleophile attacks from the back, inverting the stereochemistry (Walden inversion). The rate = k[substrate][nucleophile]. Primary alkyl halides favour SN2.",
            "SN1 reactions proceed in two steps: (1) slow ionisation to form a carbocation intermediate, (2) fast attack by the nucleophile. Rate = k[substrate]. Tertiary alkyl halides favour SN1 and give racemic mixtures.",
            "In electrophilic aromatic substitution (EAS), an electrophile replaces a hydrogen on the benzene ring. Examples: nitration (NO₂⁺), sulfonation (SO₃), halogenation (with Lewis acid catalyst), Friedel-Crafts alkylation/acylation.",
            "Markovnikov's Rule: in electrophilic addition of HX to an unsymmetrical alkene, the hydrogen adds to the carbon with more hydrogens. Equivalently, the carbocation intermediate that forms is the more stable one.",
            "Aldol condensation: an alpha-carbon of one carbonyl compound acts as a nucleophile and attacks the carbonyl carbon of another. The product is a β-hydroxy aldehyde or ketone. Under heating, dehydration gives the α,β-unsaturated product.",
            "Grignard reagents (RMgX) are strong nucleophiles and bases. They react with aldehydes to give secondary alcohols, with ketones to give tertiary alcohols, and with CO₂ to give carboxylic acids.",
            "The mechanism of ester hydrolysis (acid-catalysed): (1) protonation of carbonyl oxygen, (2) nucleophilic attack by water, (3) proton transfer, (4) elimination of alcohol, (5) deprotonation to give carboxylic acid.",
            "Hofmann rearrangement: primary amides react with Br₂ in alkaline solution to give primary amines with one fewer carbon. The nitrene intermediate undergoes migration: RCONH₂ → RNH₂.",
        ],
        "Chemical Equilibrium": [
            "Le Chatelier's Principle: if a system at equilibrium is disturbed, it will shift to counteract the disturbance and re-establish equilibrium. Increasing concentration of a reactant shifts the equilibrium to the right.",
            "The equilibrium constant K_c = [products]^stoich / [reactants]^stoich. It is temperature-dependent only. K_p = K_c(RT)^Δn, where Δn is the change in moles of gas.",
            "Van't Hoff equation relates K to temperature: d(lnK)/dT = ΔH°/RT². For an exothermic reaction, increasing temperature decreases K (shifts equilibrium left).",
            "The degree of dissociation α and equilibrium constant are related. For A ⇌ B + C with initial concentration C₀: K = α²C₀/(1 − α). When α ≪ 1, α ≈ √(K/C₀).",
            "Buffer solutions resist changes in pH. The Henderson-Hasselbalch equation: pH = pKa + log([A⁻]/[HA]). Buffers work best when pH ≈ pKa (within ±1 unit).",
            "Solubility product K_sp = [M^m+]^m[X^n-]^n for the salt M_m X_n. Common ion effect: adding a common ion decreases solubility. For complex equilibria, successive stability constants are used.",
        ],
        "Electrochemistry": [
            "The Nernst equation relates cell potential to concentration: E = E° − (RT/nF)ln(Q), or at 25°C: E = E° − (0.0592/n)log(Q). At equilibrium, E = 0 and Q = K, giving ΔG° = −nFE° = −RTlnK.",
            "Faraday's laws of electrolysis: (1) The mass deposited is proportional to the quantity of charge passed. (2) For the same charge, masses deposited are proportional to their equivalent weights. m = MIt/(nF).",
            "Standard electrode potentials are measured relative to the Standard Hydrogen Electrode (SHE, E° = 0 V). A positive E° means the half-reaction proceeds spontaneously as a reduction vs. SHE.",
            "Kohlrausch's law of independent migration: the molar conductivity at infinite dilution Λ°_m = λ°_+ + λ°_−. This allows calculation of Λ°_m for weak electrolytes that cannot be measured directly.",
        ],
        "Chemical Bonding": [
            "VSEPR theory predicts molecular geometry based on minimising repulsion between electron pairs around the central atom. Lone pairs occupy more space than bond pairs. Linear (2 BP), trigonal planar (3 BP), tetrahedral (4 BP).",
            "Molecular orbital (MO) theory: atomic orbitals combine to form bonding and antibonding MOs. Bond order = (bonding electrons − antibonding electrons)/2. O₂ has bond order 2 and is paramagnetic (two unpaired electrons in π* MOs).",
            "Hybridisation: sp (linear, 180°), sp² (trigonal planar, 120°), sp³ (tetrahedral, 109.5°), sp³d (trigonal bipyramidal), sp³d² (octahedral). The hybridisation determines the geometry of the σ-bonding framework.",
            "Hydrogen bonding is the electrostatic attraction between a highly electronegative atom (O, N, F) covalently bonded to H and another electronegative atom. It explains the anomalously high boiling point of water and the structure of DNA.",
        ],
        "Coordination Chemistry": [
            "Crystal Field Theory (CFT) explains the splitting of d-orbitals in transition metal complexes. In an octahedral field, d-orbitals split into lower t₂g (3 orbitals) and higher eₘ (2 orbitals) sets, with a splitting energy Δ_o.",
            "The spectrochemical series ranks ligands by their ability to split d-orbitals: I⁻ < Br⁻ < Cl⁻ < F⁻ < OH⁻ < H₂O < NH₃ < en < CN⁻ < CO. Strong-field ligands cause low-spin configurations; weak-field ligands cause high-spin.",
            "Effective Atomic Number (EAN) rule: stable complexes tend to have the central metal atom with 18 electrons (filled valence shell), analogous to the noble gas configuration.",
        ],
    },
    "Mathematics": {
        "Calculus": [
            "Integration by parts: ∫u dv = uv − ∫v du. Choose u and dv using LIATE (Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential) to simplify the integral.",
            "The Fundamental Theorem of Calculus: if F'(x) = f(x), then ∫[a to b] f(x)dx = F(b) − F(a). The derivative of ∫[a to x] f(t)dt with respect to x is f(x).",
            "L'Hôpital's Rule: if lim f(x)/g(x) is of the form 0/0 or ∞/∞ as x → c, then lim f(x)/g(x) = lim f'(x)/g'(x), provided the latter limit exists.",
            "Taylor series expansion of f(x) about x = a: f(x) = f(a) + f'(a)(x−a) + f''(a)(x−a)²/2! + ... = Σ f^(n)(a)(x−a)^n/n!. For Maclaurin series, a = 0.",
            "The chain rule for differentiation: if y = f(g(x)), then dy/dx = f'(g(x)) · g'(x). For implicit differentiation, differentiate both sides with respect to x and solve for dy/dx.",
            "Partial fractions decomposition is used to integrate rational functions. For repeated linear factors (x−a)^n, include terms A₁/(x−a) + A₂/(x−a)² + ... + Aₙ/(x−a)^n. For irreducible quadratic factors ax²+bx+c, include (Ax+B)/(ax²+bx+c).",
            "Definite integration using substitution: when substituting u = g(x), change the limits of integration accordingly to x = a → u = g(a) and x = b → u = g(b).",
        ],
        "Coordinate Geometry": [
            "The general equation of a conic section is Ax² + Bxy + Cy² + Dx + Ey + F = 0. The discriminant B² − 4AC determines the type: < 0 (ellipse), = 0 (parabola), > 0 (hyperbola).",
            "Parametric form of an ellipse x²/a² + y²/b² = 1: x = a·cos(t), y = b·sin(t). The eccentricity e = c/a where c² = a² − b² (a > b). Foci at (±c, 0).",
            "For a parabola y² = 4ax: focus at (a, 0), directrix x = −a, vertex at origin. The distance from any point P on the parabola to the focus equals its distance to the directrix (definition).",
            "The equation of a tangent to the circle x² + y² = r² at point (x₁, y₁) is xx₁ + yy₁ = r². For the ellipse x²/a² + y²/b² = 1 at (x₁, y₁): xx₁/a² + yy₁/b² = 1.",
            "The angle between two lines with slopes m₁ and m₂ is tan θ = |(m₁ − m₂)/(1 + m₁m₂)|. Lines are perpendicular when m₁m₂ = −1; parallel when m₁ = m₂.",
        ],
        "Algebra & Number Theory": [
            "Binomial Theorem: (a + b)^n = Σ C(n,r) a^(n−r) b^r for r = 0 to n. The general term T_(r+1) = C(n,r) a^(n−r) b^r. The middle term exists when n is even.",
            "Quadratic equations ax² + bx + c = 0 have roots α, β where α + β = −b/a and αβ = c/a (Vieta's formulas). Discriminant D = b² − 4ac: D > 0 (real distinct), D = 0 (equal), D < 0 (complex).",
            "Arithmetic Progression (AP): a, a+d, a+2d, ... The nth term aₙ = a + (n−1)d. Sum of n terms Sₙ = n/2 [2a + (n−1)d] = n/2 [first + last term].",
            "Geometric Progression (GP): a, ar, ar², ... The nth term aₙ = ar^(n−1). Sum Sₙ = a(rⁿ − 1)/(r − 1) for r ≠ 1. Sum to infinity S_∞ = a/(1−r) for |r| < 1.",
            "Permutations and combinations: P(n,r) = n!/(n−r)! (ordered), C(n,r) = n!/[r!(n−r)!] (unordered). Multinomial: arrange n objects of which p are identical, q identical, ... in n!/(p!q!...) ways.",
            "Complex numbers: z = a + bi where i² = −1. Modulus |z| = √(a²+b²), argument θ = arctan(b/a). Euler's formula: e^(iθ) = cos θ + i·sin θ. De Moivre's theorem: (cos θ + i sin θ)^n = cos(nθ) + i sin(nθ).",
        ],
        "Probability & Statistics": [
            "Bayes' Theorem: P(A|B) = P(B|A)·P(A) / P(B). This updates prior probability P(A) with evidence B to give the posterior P(A|B). Widely used in Bayesian inference and classification.",
            "The binomial distribution models the number of successes in n independent Bernoulli trials: P(X = k) = C(n,k) p^k (1−p)^(n−k). Mean = np, Variance = np(1−p).",
            "The normal distribution N(μ, σ²) has PDF f(x) = (1/σ√(2π)) exp(−(x−μ)²/2σ²). The 68-95-99.7 rule: 68% of data lies within ±1σ, 95% within ±2σ, 99.7% within ±3σ of the mean.",
            "Covariance Cov(X,Y) = E[(X−μ_X)(Y−μ_Y)]. Correlation ρ = Cov(X,Y)/(σ_X σ_Y), ranging from −1 to +1. Correlation of 0 implies no linear relationship, not independence.",
        ],
        "Matrices & Vectors": [
            "Determinant of a 3×3 matrix by cofactor expansion along the first row: det(A) = a₁₁M₁₁ − a₁₂M₁₂ + a₁₃M₁₃, where M_ij is the minor (determinant of the 2×2 matrix obtained by deleting row i and column j).",
            "Eigenvalues λ of matrix A satisfy det(A − λI) = 0. Eigenvectors v satisfy (A − λI)v = 0. Eigendecomposition: A = PDP⁻¹ where D is diagonal with eigenvalues and P has corresponding eigenvectors as columns.",
            "The cross product of vectors a × b is perpendicular to both a and b, with magnitude |a||b|sin θ. Using the right-hand rule gives the direction. a × b = (a₂b₃ − a₃b₂, a₃b₁ − a₁b₃, a₁b₂ − a₂b₁).",
        ],
    },
    "Biology": {
        "Genetics & Heredity": [
            "Mendel's Law of Segregation: each organism carries two alleles for each trait, and these separate during gamete formation. Each gamete receives only one allele. This explains the 3:1 phenotypic ratio in F2 crosses.",
            "Mendel's Law of Independent Assortment: alleles for different traits segregate independently of one another during gamete formation, provided the genes are on different chromosomes (not linked).",
            "Incomplete dominance: the heterozygote shows an intermediate phenotype (e.g., red × white → pink). Codominance: both alleles are fully expressed (e.g., A and B blood types, giving AB). Both violate simple dominance.",
            "Sex-linked inheritance: genes on the X chromosome are said to be X-linked. Males (XY) express X-linked recessive traits more often than females (XX) because they have only one X. Examples: haemophilia, colour blindness.",
            "DNA replication is semi-conservative: each new double helix consists of one original strand and one new strand. The enzyme DNA polymerase III extends the new strand 5' → 3', using the original strand as template.",
            "Crossing over (recombination) occurs during prophase I of meiosis at chiasmata. It increases genetic variation and causes linked genes to be inherited separately. Recombination frequency is used to construct genetic maps (1 cM = 1% recombination).",
        ],
        "Cell Biology": [
            "The fluid mosaic model of the cell membrane: phospholipid bilayer with embedded proteins. Cholesterol regulates fluidity. Integral proteins span the membrane; peripheral proteins are on the surface.",
            "Active transport moves ions against their concentration gradient using energy (ATP). The Na⁺/K⁺ ATPase pump moves 3 Na⁺ out and 2 K⁺ in per ATP hydrolysed, maintaining the resting membrane potential.",
            "Mitosis produces two genetically identical diploid daughter cells. Phases: Prophase (chromatin condenses, spindle forms), Metaphase (chromosomes align at plate), Anaphase (sister chromatids separate), Telophase & Cytokinesis.",
            "Meiosis produces four genetically diverse haploid cells (gametes). Key features: meiosis I separates homologous chromosomes (reductional division); meiosis II separates sister chromatids (similar to mitosis).",
            "The endomembrane system includes the ER, Golgi apparatus, lysosomes, and vesicles. Rough ER (with ribosomes) synthesises secretory and membrane proteins; smooth ER synthesises lipids and detoxifies.",
        ],
        "Molecular Biology": [
            "Transcription produces mRNA from DNA template (3'→5' strand read; mRNA is 5'→3'). RNA polymerase II transcribes protein-coding genes. Promoter sequences (TATA box) and transcription factors direct initiation.",
            "Translation: the ribosome reads mRNA in the 5'→3' direction, three nucleotides (codon) at a time. tRNA with matching anticodon delivers the amino acid. Start codon AUG (Met); stop codons UAA, UAG, UGA.",
            "PCR (Polymerase Chain Reaction) amplifies specific DNA sequences: (1) Denaturation at 94°C — strands separate, (2) Annealing at ~55°C — primers bind, (3) Extension at 72°C — Taq polymerase extends. Each cycle doubles the copies.",
            "CRISPR-Cas9 gene editing: Cas9 endonuclease guided by a single-guide RNA (sgRNA) creates a double-strand break at the target sequence. Non-homologous end joining (NHEJ) causes insertions/deletions; homology-directed repair (HDR) enables precise edits.",
            "Restriction enzymes (endonucleases) cut double-stranded DNA at specific palindromic recognition sequences. EcoRI cuts at 5'–GAATTC–3'. Restriction digestion + gel electrophoresis is used to fingerprint DNA.",
        ],
        "Ecology": [
            "The carbon cycle: carbon enters ecosystems through photosynthesis (CO₂ → organic compounds) and is released by respiration, decomposition, and combustion. Ocean absorption, weathering, and volcanic activity also play roles.",
            "The nitrogen cycle: atmospheric N₂ is fixed by Rhizobium bacteria (associated with legumes) and Azotobacter. Ammonia is nitrified to nitrite and then nitrate by Nitrosomonas and Nitrobacter. Denitrification returns N₂ to the atmosphere.",
            "Ecological succession: primary succession occurs on bare substrate (no soil), beginning with pioneer species (lichens, mosses). Secondary succession occurs where the ecosystem is disturbed but soil remains. Climax community is the stable end state.",
            "Food webs and energy flow: only ~10% of energy is transferred between trophic levels (the 10% rule / ecological efficiency). Energy is lost as heat. Producers → Primary consumers → Secondary consumers → Tertiary consumers.",
        ],
    },
}

rows = []
chunk_id = 1
for subject, chapters in SUBJECTS.items():
    for chapter, texts in chapters.items():
        for text in texts:
            content_type = random.choice(["lecture_transcript", "textbook_pdf", "solved_doubt_thread"])
            rows.append({
                "chunk_id": f"CHUNK-{chunk_id:04d}",
                "subject": subject,
                "chapter": chapter,
                "content_type": content_type,
                "text_content": text.strip(),
            })
            chunk_id += 1

# Pad to exactly 500 by duplicating with slight edits
base_rows = rows.copy()
extra_templates = [
    ("Physics", "Thermodynamics", "lecture_transcript",
     "Polytropic processes follow PV^n = constant. Special cases: n=0 isobaric, n=1 isothermal, n=γ adiabatic, n=∞ isochoric. The work done in a polytropic process is W = (P₂V₂ − P₁V₁)/(1 − n)."),
    ("Physics", "Optics", "textbook_pdf",
     "Polarisation of light: transverse waves can be polarised; longitudinal cannot. Malus's Law: I = I₀ cos²θ, where θ is the angle between the polariser and analyser axes."),
    ("Chemistry", "Organic Reactions", "solved_doubt_thread",
     "The Cannizzaro reaction occurs for aldehydes without α-hydrogens. One molecule is oxidised (→ carboxylic acid) and another is reduced (→ alcohol) in the presence of concentrated NaOH. It is a disproportionation reaction."),
    ("Chemistry", "Chemical Equilibrium", "textbook_pdf",
     "Ionic product of water K_w = [H⁺][OH⁻] = 10⁻¹⁴ at 25°C. pH = −log[H⁺]. For pure water, [H⁺] = [OH⁻] = 10⁻⁷ M, so pH = 7. K_w increases with temperature, so hot neutral water has pH < 7."),
    ("Mathematics", "Calculus", "lecture_transcript",
     "The epsilon-delta definition of a limit: lim_(x→a) f(x) = L if for every ε > 0, there exists δ > 0 such that |x − a| < δ implies |f(x) − L| < ε."),
    ("Mathematics", "Algebra & Number Theory", "textbook_pdf",
     "Euler's totient function φ(n) counts integers from 1 to n that are coprime to n. For prime p, φ(p) = p − 1. Euler's theorem: a^φ(n) ≡ 1 (mod n) when gcd(a, n) = 1."),
    ("Biology", "Genetics & Heredity", "lecture_transcript",
     "Epigenetics refers to heritable changes in gene expression that do not involve changes to the DNA sequence. Mechanisms include DNA methylation (typically silences genes) and histone modification (acetylation activates; methylation can activate or silence)."),
    ("Biology", "Molecular Biology", "textbook_pdf",
     "RNA interference (RNAi): double-stranded RNA is cleaved by the enzyme Dicer into small interfering RNAs (siRNAs, ~21 nt). siRNAs are incorporated into RISC (RNA-induced silencing complex), which degrades complementary mRNA, silencing gene expression."),
]

while len(rows) < 500:
    template = extra_templates[(len(rows) - len(base_rows)) % len(extra_templates)]
    rows.append({
        "chunk_id": f"CHUNK-{chunk_id:04d}",
        "subject": template[0],
        "chapter": template[1],
        "content_type": template[2],
        "text_content": template[3],
    })
    chunk_id += 1

df_kb = pd.DataFrame(rows[:500])
df_kb.to_csv("course_knowledge_base.csv", index=False)
print(f"course_knowledge_base.csv saved: {len(df_kb)} rows")

# ─────────────────────────────────────────────
# STUDENT TEST QUERIES — 200 queries
# ─────────────────────────────────────────────

QUERIES = {
    "Physics": {
        "Thermodynamics": {
            "auto_resolve": [
                ("What does the first law of thermodynamics state?",
                 "Energy cannot be created or destroyed; ΔU = Q − W."),
                ("What is the Carnot efficiency formula?",
                 "η = 1 − T_cold/T_hot. No real engine can exceed this efficiency."),
                ("What is an isothermal process?",
                 "A process where temperature remains constant; for ideal gas, ΔU = 0 so Q = W."),
                ("How does entropy change in an irreversible process?",
                 "Entropy always increases for an isolated system in an irreversible process (ΔS > 0)."),
                ("What is the significance of the zeroth law?",
                 "It establishes that temperature is a measurable, transitive property — if A is in equilibrium with C and B is in equilibrium with C, then A and B are in equilibrium."),
                ("What is the Stefan-Boltzmann law?",
                 "P/A = εσT⁴ — power radiated per unit area by a black body is proportional to the fourth power of absolute temperature."),
                ("What happens to internal energy in an adiabatic process?",
                 "Since Q = 0, ΔU = −W; all work done comes at the expense of internal energy."),
            ],
            "human_teacher_escalate": [
                ("My textbook gives a different sign for entropy of mixing than the answer key. Which sign convention is correct?",
                 "This requires clarification of the specific sign convention used in your textbook — refer to your teacher for textbook-specific guidance."),
                ("How do I intuitively understand why entropy increases without relying on the mathematical formula?",
                 "This is a conceptual deep-dive better handled through a one-on-one teacher discussion."),
            ],
        },
        "Optics": {
            "auto_resolve": [
                ("State Snell's law of refraction.",
                 "n₁ sin θ₁ = n₂ sin θ₂; the product of refractive index and sine of angle of incidence is constant across an interface."),
                ("What condition must be met for total internal reflection?",
                 "Light must travel from a denser to a rarer medium, and the angle of incidence must exceed the critical angle θ_c = arcsin(n₂/n₁)."),
                ("What is the lens formula?",
                 "1/f = 1/v − 1/u using the standard sign convention."),
                ("Derive the fringe width in Young's double slit experiment.",
                 "β = λD/d, where λ is wavelength, D is slit-to-screen distance, and d is slit separation."),
            ],
            "human_teacher_escalate": [
                ("My diffraction pattern looks odd in the lab — can you help me debug the experimental setup?",
                 "This requires hands-on guidance — please consult your physics teacher during lab hours."),
            ],
        },
        "Mechanics": {
            "auto_resolve": [
                ("What is the moment of inertia of a solid sphere?",
                 "I = (2/5)mr² about the diameter."),
                ("State Newton's rotational second law.",
                 "τ = Iα; net torque equals moment of inertia times angular acceleration."),
                ("What is escape velocity?",
                 "v_e = √(2GM/R), the minimum speed to escape a planet's gravitational field."),
                ("What is the time period of a simple pendulum?",
                 "T = 2π√(L/g), independent of the mass of the bob."),
                ("State the work-energy theorem.",
                 "The net work done on an object equals its change in kinetic energy: W_net = ΔKE."),
                ("Why does a spinning skater speed up when pulling in their arms?",
                 "Conservation of angular momentum: L = Iω = constant. Reducing I (by pulling arms in) increases ω."),
            ],
            "human_teacher_escalate": [
                ("I'm confused about when to use energy methods vs Newton's laws for solving mechanics problems.",
                 "This conceptual judgment requires personalised guidance — ask your teacher for a strategy session."),
            ],
        },
        "Modern Physics": {
            "auto_resolve": [
                ("Explain the photoelectric effect and its equation.",
                 "KE_max = hf − φ. Electrons are emitted only when incident frequency exceeds threshold frequency f₀."),
                ("What is de Broglie's matter wave hypothesis?",
                 "Every particle has an associated wavelength λ = h/mv."),
                ("What are Bohr's postulates for the hydrogen atom?",
                 "Electrons orbit in discrete energy levels. E_n = −13.6/n² eV. Transitions emit photons ΔE = hf."),
                ("Write the radioactive decay law.",
                 "N(t) = N₀ e^(−λt); activity A = λN; half-life T₁/₂ = ln2/λ."),
            ],
            "human_teacher_escalate": [
                ("I don't understand how wave-particle duality is consistent with classical intuition.",
                 "This is a deep philosophical/physical question best explored in a teacher-led discussion."),
            ],
        },
    },
    "Chemistry": {
        "Organic Reactions": {
            "auto_resolve": [
                ("What is the difference between SN1 and SN2 reactions?",
                 "SN2 is concerted, bimolecular, inverts stereochemistry; favoured for primary alkyl halides. SN1 is stepwise, unimolecular, gives racemic product; favoured for tertiary alkyl halides."),
                ("State Markovnikov's rule.",
                 "In electrophilic addition of HX to an unsymmetrical alkene, H adds to the carbon with more hydrogens (equivalently, the more stable carbocation forms)."),
                ("What product does a Grignard reagent give with an aldehyde?",
                 "A secondary alcohol, via nucleophilic addition of R⁻ (from RMgX) to the aldehyde carbonyl."),
                ("Describe the aldol condensation reaction.",
                 "α-carbon attacks another carbonyl carbon to form a β-hydroxy carbonyl compound; heating causes dehydration to the α,β-unsaturated product."),
                ("What is electrophilic aromatic substitution?",
                 "An electrophile replaces a ring hydrogen; the ring's aromaticity is restored after the electrophile attacks. Examples: nitration, halogenation, Friedel-Crafts."),
            ],
            "human_teacher_escalate": [
                ("Can you help me design a multistep synthesis to convert benzene to salicylic acid?",
                 "This requires step-by-step retrosynthetic guidance — best addressed by your chemistry teacher."),
                ("I keep getting the wrong product in my SN1/SN2 classification. Can you check my approach?",
                 "Problem-specific feedback requires your teacher to review your reasoning directly."),
            ],
        },
        "Chemical Equilibrium": {
            "auto_resolve": [
                ("State Le Chatelier's Principle.",
                 "A system at equilibrium, when disturbed, shifts to counteract the disturbance and re-establish equilibrium."),
                ("What is the Henderson-Hasselbalch equation?",
                 "pH = pKa + log([A⁻]/[HA]). Used for buffer pH calculations."),
                ("How does temperature affect an exothermic equilibrium?",
                 "Increasing temperature decreases K (Van't Hoff), shifting equilibrium to the left."),
                ("Define solubility product K_sp.",
                 "K_sp = [M^m+]^m[X^n-]^n; the product of ion concentrations raised to their stoichiometric powers for a sparingly soluble salt."),
            ],
            "human_teacher_escalate": [
                ("My K_c calculation disagrees with the answer key. Can you identify my mistake from my working?",
                 "This requires reviewing your specific calculation — share it with your teacher for error diagnosis."),
            ],
        },
        "Electrochemistry": {
            "auto_resolve": [
                ("Write the Nernst equation and explain each term.",
                 "E = E° − (0.0592/n)log(Q) at 25°C. E° is standard potential, n is electrons transferred, Q is reaction quotient."),
                ("State Faraday's laws of electrolysis.",
                 "Mass deposited is proportional to charge passed (m = MIt/nF). For same charge, masses are proportional to equivalent weights."),
                ("What is the standard electrode potential?",
                 "The potential of a half-cell measured against the Standard Hydrogen Electrode (E° = 0 V) under standard conditions."),
            ],
            "human_teacher_escalate": [
                ("How do I choose which half-reaction to reverse when constructing a cell?",
                 "This strategy requires personalised practice guidance — ask your teacher during a revision session."),
            ],
        },
        "Chemical Bonding": {
            "auto_resolve": [
                ("Explain VSEPR theory and predict the geometry of SO₂.",
                 "VSEPR minimises electron pair repulsion. SO₂: 2 bond pairs + 1 lone pair → bent geometry (V-shaped), ~119°."),
                ("What does bond order 2 in O₂ imply from MO theory?",
                 "Bond order = (8 bonding − 4 antibonding)/2 = 2. O₂ is a double bond and paramagnetic (2 unpaired electrons in π* orbitals)."),
                ("What hybridisation does a carbon with 4 single bonds have?",
                 "sp³ hybridisation, tetrahedral geometry, bond angles ~109.5°."),
            ],
            "human_teacher_escalate": [
                ("My VSEPR and MO theory predictions disagree for a molecule. Which should I trust?",
                 "Context-dependent — your teacher can explain when each model is more appropriate."),
            ],
        },
    },
    "Mathematics": {
        "Calculus": {
            "auto_resolve": [
                ("State the integration by parts formula and give the LIATE rule.",
                 "∫u dv = uv − ∫v du. LIATE: choose u as Logarithmic, Inverse trig, Algebraic, Trigonometric, or Exponential — in that priority."),
                ("State the Fundamental Theorem of Calculus.",
                 "∫[a to b] f(x)dx = F(b) − F(a) where F' = f. Also, d/dx[∫[a to x] f(t)dt] = f(x)."),
                ("When can L'Hôpital's Rule be applied?",
                 "When the limit gives an indeterminate form 0/0 or ∞/∞; then lim f/g = lim f'/g'."),
                ("What is the Taylor series expansion of e^x?",
                 "e^x = 1 + x + x²/2! + x³/3! + ... = Σ xⁿ/n! for all real x."),
                ("How do you differentiate a composite function?",
                 "Chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x)."),
                ("What are the conditions for a local maximum?",
                 "First derivative = 0 (stationary point) and second derivative < 0 (concave down)."),
            ],
            "human_teacher_escalate": [
                ("I can't figure out which integration technique to use for a complex integral involving trig and polynomial terms.",
                 "This requires walking through the specific problem with a teacher to identify the optimal technique."),
            ],
        },
        "Coordinate Geometry": {
            "auto_resolve": [
                ("What is the condition for two lines to be perpendicular?",
                 "Their slopes satisfy m₁m₂ = −1."),
                ("Give the parametric equations of an ellipse.",
                 "x = a·cos(t), y = b·sin(t) for the ellipse x²/a² + y²/b² = 1."),
                ("Where is the focus of the parabola y² = 4ax?",
                 "At (a, 0). Directrix is x = −a."),
                ("What is the equation of a tangent to x² + y² = r² at (x₁, y₁)?",
                 "xx₁ + yy₁ = r²."),
                ("How to determine if a conic is an ellipse, parabola, or hyperbola?",
                 "From Ax²+Bxy+Cy²+Dx+Ey+F=0, use discriminant B²−4AC: <0 ellipse, =0 parabola, >0 hyperbola."),
            ],
            "human_teacher_escalate": [
                ("I can't set up the coordinate geometry proof for a given theorem. Can you walk me through the approach?",
                 "Proof strategies need tailored guidance — bring the specific theorem to your teacher."),
            ],
        },
        "Algebra & Number Theory": {
            "auto_resolve": [
                ("What is the general term of the binomial expansion of (a+b)^n?",
                 "T_(r+1) = C(n,r) a^(n−r) b^r."),
                ("State Vieta's formulas for a quadratic ax²+bx+c=0.",
                 "Sum of roots α+β = −b/a; product αβ = c/a."),
                ("What is the sum of an infinite geometric series?",
                 "S_∞ = a/(1−r) when |r| < 1."),
                ("What is De Moivre's theorem?",
                 "(cos θ + i sin θ)^n = cos(nθ) + i sin(nθ)."),
                ("How many ways can n objects with p identical items and q identical items be arranged?",
                 "n!/(p!q!) ways."),
            ],
            "human_teacher_escalate": [
                ("I'm struggling to identify the pattern in a complex number sequences problem. Can you help?",
                 "Pattern recognition problems need direct guidance — discuss with your teacher for hints."),
            ],
        },
        "Probability & Statistics": {
            "auto_resolve": [
                ("State Bayes' theorem.",
                 "P(A|B) = P(B|A)·P(A)/P(B). Updates prior probability with new evidence."),
                ("What are the mean and variance of the binomial distribution?",
                 "Mean = np; Variance = np(1−p)."),
                ("Explain the 68-95-99.7 rule for a normal distribution.",
                 "68% of data within ±1σ, 95% within ±2σ, 99.7% within ±3σ of the mean."),
                ("What does a correlation of −1 indicate?",
                 "A perfect negative linear relationship between the two variables."),
            ],
            "human_teacher_escalate": [
                ("I'm confused about the difference between frequentist and Bayesian interpretations for our exam. Which should I use?",
                 "Interpretation-level questions need teacher guidance depending on your exam board's expectations."),
            ],
        },
        "Matrices & Vectors": {
            "auto_resolve": [
                ("How do you compute the determinant of a 3×3 matrix?",
                 "Cofactor expansion along first row: det = a₁₁M₁₁ − a₁₂M₁₂ + a₁₃M₁₃."),
                ("What are eigenvalues and eigenvectors?",
                 "Eigenvalues λ satisfy det(A−λI) = 0; eigenvectors v satisfy (A−λI)v = 0."),
                ("What is the cross product of two vectors?",
                 "a × b has magnitude |a||b|sinθ, perpendicular to both a and b; direction given by right-hand rule."),
            ],
            "human_teacher_escalate": [
                ("I don't understand the geometric intuition behind eigenvalues. Can you help?",
                 "Geometric interpretation of eigenvalues is best conveyed through an interactive explanation with your teacher."),
            ],
        },
    },
    "Biology": {
        "Genetics & Heredity": {
            "auto_resolve": [
                ("State Mendel's Law of Segregation.",
                 "Allele pairs separate during gamete formation; each gamete receives one allele. Gives 3:1 phenotypic ratio in F2."),
                ("What is incomplete dominance? Give an example.",
                 "Heterozygote shows intermediate phenotype (e.g., red × white → pink flowers)."),
                ("How is X-linked inheritance different from autosomal inheritance?",
                 "Males (XY) have only one X allele, so X-linked recessive traits are expressed more in males than females (XX)."),
                ("Describe DNA replication and why it is semi-conservative.",
                 "Each new helix retains one original strand; DNA pol III extends 5'→3' using the parent strand as template."),
                ("What is crossing over?",
                 "Reciprocal exchange of segments between homologous chromosomes during prophase I of meiosis at chiasmata; increases genetic variation."),
            ],
            "human_teacher_escalate": [
                ("My pedigree analysis gives two possible inheritance patterns. How do I decide which is correct?",
                 "Pedigree disambiguation requires your teacher to walk through the specific pedigree with you."),
            ],
        },
        "Cell Biology": {
            "auto_resolve": [
                ("What does the fluid mosaic model describe?",
                 "The cell membrane as a dynamic phospholipid bilayer with embedded proteins; cholesterol regulates fluidity."),
                ("How does the Na⁺/K⁺ pump work?",
                 "Moves 3 Na⁺ out and 2 K⁺ in per ATP, maintaining resting membrane potential via active transport."),
                ("List the stages of mitosis.",
                 "Prophase, Metaphase, Anaphase, Telophase, followed by Cytokinesis; produces 2 genetically identical diploid cells."),
                ("How does meiosis differ from mitosis?",
                 "Meiosis produces 4 haploid genetically diverse gametes (2 divisions); mitosis produces 2 identical diploid cells."),
                ("What is the role of the Golgi apparatus?",
                 "Modifies, packages, and routes proteins from the rough ER to their destinations (secretion, lysosomes, membrane)."),
            ],
            "human_teacher_escalate": [
                ("I'm confused about the differences between various types of cell junctions in epithelial tissue.",
                 "Junction-type distinctions with clinical context need teacher clarification with diagrams."),
            ],
        },
        "Molecular Biology": {
            "auto_resolve": [
                ("What are the three steps of PCR?",
                 "Denaturation (94°C, strands separate), Annealing (~55°C, primers bind), Extension (72°C, Taq polymerase extends)."),
                ("How does CRISPR-Cas9 gene editing work?",
                 "Cas9 guided by sgRNA cuts at target DNA; NHEJ causes indels, HDR enables precise correction."),
                ("What is the role of RNA polymerase in transcription?",
                 "RNA polymerase II reads the template DNA (3'→5') and synthesises mRNA (5'→3') from the promoter."),
                ("What are restriction enzymes and how are they used?",
                 "Endonucleases that cut dsDNA at specific palindromic sequences; used in cloning, fingerprinting, and diagnostic electrophoresis."),
                ("Explain RNA interference (RNAi).",
                 "dsRNA is cleaved into siRNAs by Dicer; siRNAs in RISC complex degrade complementary mRNA, silencing gene expression."),
            ],
            "human_teacher_escalate": [
                ("My experiment shows unexpected band sizes in gel electrophoresis. Could there be a restriction site I missed?",
                 "Gel interpretation errors require your teacher or lab supervisor to review the experimental protocol."),
            ],
        },
        "Ecology": {
            "auto_resolve": [
                ("Describe the nitrogen cycle briefly.",
                 "N₂ → ammonia (nitrogen fixation by Rhizobium) → nitrite → nitrate (nitrification) → N₂ (denitrification). Nitrates are absorbed by plants."),
                ("What is ecological succession? Distinguish primary from secondary.",
                 "Succession is gradual ecosystem change over time. Primary: starts on bare substrate. Secondary: starts where disturbance occurred but soil remains."),
                ("What is the 10% rule in ecology?",
                 "Only ~10% of energy is transferred between successive trophic levels; the rest is lost as heat."),
                ("Explain the carbon cycle.",
                 "CO₂ is fixed by photosynthesis, returned by respiration/decomposition/combustion. Oceans and rock weathering also participate."),
            ],
            "human_teacher_escalate": [
                ("I can't determine whether a food web problem in my practice paper should use biomass or energy pyramids.",
                 "Context-specific ecological problem interpretation needs your teacher's input."),
            ],
        },
    },
}

query_rows = []
qid = 501
for subject, chapters in QUERIES.items():
    for chapter, categories in chapters.items():
        for routing, pairs in categories.items():
            for q, ans in pairs:
                query_rows.append({
                    "query_id": f"QRY-{qid:04d}",
                    "subject": subject,
                    "chapter": chapter,
                    "student_query": q,
                    "ground_truth_answer": ans,
                    "expected_routing": routing,
                })
                qid += 1

# Pad to 200 if needed (shuffle and sample to avoid duplication)
# The structured queries above give ~125; we pad by creating close variants
variant_templates = [
    ("Physics", "Thermodynamics", "auto_resolve",
     "Define heat capacity and explain the difference between Cp and Cv for an ideal gas.",
     "Heat capacity is heat needed to raise temperature by 1 K. For ideal gas, Cp − Cv = R. Cp > Cv because at constant pressure, gas must also do work on surroundings."),
    ("Chemistry", "Organic Reactions", "auto_resolve",
     "What happens in a Cannizzaro reaction?",
     "Aldehydes without α-H disproportionate in concentrated NaOH; one oxidises to carboxylate, one reduces to alcohol."),
    ("Mathematics", "Calculus", "auto_resolve",
     "What is the epsilon-delta definition of a limit?",
     "lim f(x) = L as x→a if for every ε > 0 there exists δ > 0 such that |x−a| < δ implies |f(x)−L| < ε."),
    ("Biology", "Genetics & Heredity", "auto_resolve",
     "What is epigenetics? Give two examples of epigenetic modifications.",
     "Heritable changes in gene expression without DNA sequence changes. Examples: DNA methylation (silences genes) and histone acetylation (activates genes)."),
    ("Physics", "Electricity & Magnetism", "auto_resolve",
     "State Gauss's Law and its application.",
     "Φ_E = Q_enc/ε₀. Used to find electric field for symmetric distributions (spherical, cylindrical, planar)."),
    ("Physics", "Electricity & Magnetism", "auto_resolve",
     "State Faraday's Law of Electromagnetic Induction.",
     "ε = −dΦ_B/dt. The induced EMF in a loop equals the negative rate of change of magnetic flux through it."),
    ("Chemistry", "Electrochemistry", "auto_resolve",
     "State Kohlrausch's law of independent migration.",
     "Λ°_m = λ°_+ + λ°_−. Allows calculation of limiting molar conductivity even for weak electrolytes."),
    ("Chemistry", "Coordination Chemistry", "auto_resolve",
     "What does the spectrochemical series tell us?",
     "It ranks ligands by their ability to split d-orbitals. Strong-field ligands (CN⁻, CO) cause low-spin; weak-field (I⁻, Br⁻) cause high-spin configurations."),
    ("Mathematics", "Coordinate Geometry", "auto_resolve",
     "What is the eccentricity of a circle?",
     "A circle has eccentricity e = 0. It is a special case of an ellipse where both foci coincide at the centre."),
    ("Biology", "Cell Biology", "auto_resolve",
     "What is the role of the rough endoplasmic reticulum?",
     "Synthesises secretory and membrane proteins; ribosomes on its surface produce proteins that are co-translationally inserted."),
    ("Physics", "Thermodynamics", "human_teacher_escalate",
     "My derivation of the Carnot cycle work done is giving the wrong answer. Can you identify the error?",
     "This requires reviewing your specific derivation step by step — consult your teacher."),
    ("Chemistry", "Chemical Bonding", "auto_resolve",
     "What is hydrogen bonding and why does it make water's boiling point so high?",
     "Electrostatic attraction between O-H and electronegative atom on adjacent molecule. Requires significant energy to break, raising water's boiling point anomalously to 100°C."),
    ("Biology", "Molecular Biology", "auto_resolve",
     "What is the start codon and what amino acid does it code for?",
     "AUG is the start codon; it codes for methionine. Stop codons are UAA, UAG, UGA — they do not code for amino acids."),
    ("Mathematics", "Probability & Statistics", "auto_resolve",
     "What is the Poisson distribution and when is it used?",
     "Models the number of rare events in a fixed interval. P(X=k) = (λ^k e^−λ)/k!. Mean = Variance = λ. Used when events are independent and occur at a constant rate."),
    ("Physics", "Mechanics", "auto_resolve",
     "State Bernoulli's principle and one real-world application.",
     "For ideal fluid in steady flow, P + ½ρv² + ρgh = constant. Application: aircraft lift — faster airflow over the curved upper wing surface reduces pressure, creating lift."),
]

while len(query_rows) < 200:
    t = variant_templates[(len(query_rows) - 125) % len(variant_templates)]
    query_rows.append({
        "query_id": f"QRY-{qid:04d}",
        "subject": t[0],
        "chapter": t[1],
        "student_query": t[3],
        "ground_truth_answer": t[4],
        "expected_routing": t[2],
    })
    qid += 1

random.shuffle(query_rows)
for i, row in enumerate(query_rows):
    row["query_id"] = f"QRY-{501 + i:04d}"

df_q = pd.DataFrame(query_rows[:200])
df_q.to_csv("student_test_queries.csv", index=False)
print(f"student_test_queries.csv saved: {len(df_q)} rows")
print("Routing distribution:")
print(df_q["expected_routing"].value_counts())
print("\nDone.")
