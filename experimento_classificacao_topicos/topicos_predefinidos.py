"""Catálogo editável: conhecimento prévio, sem extrair sementes dos rótulos do CSV."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_loader import AAPD_ARXIV_CATEGORIES

# Identificador: (descrição em inglês, palavras/expressões-semente).
CATALOG = {
 'cmp-lg': ('Computation and language', 'linguistics, parsing, translation, language'),
 'cond-mat.dis-nn': ('Disordered systems and neural networks', 'disorder, spin glass, neural, networks'),
 'cond-mat.stat-mech': ('Statistical mechanics', 'statistical mechanics, entropy, phase transition, lattice'),
 'cs.AI': ('Artificial intelligence', 'reasoning, intelligence, planning, knowledge'),
 'cs.CC': ('Computational complexity', 'complexity, hardness, polynomial, completeness'),
 'cs.CE': ('Computational engineering finance and science', 'simulation, engineering, finance, computational'),
 'cs.CG': ('Computational geometry', 'geometry, triangulation, polygon, convex hull'),
 'cs.CL': ('Computational linguistics and natural language processing', 'language, parsing, translation, linguistics'),
 'cs.CR': ('Cryptography and security', 'cryptography, encryption, security, privacy'),
 'cs.CV': ('Computer vision and pattern recognition', 'image, vision, recognition, segmentation'),
 'cs.CY': ('Computers and society', 'society, ethics, policy, social'),
 'cs.DB': ('Databases', 'database, query, relational, transactions'),
 'cs.DC': ('Distributed parallel and cluster computing', 'distributed, parallel, synchronization, consensus'),
 'cs.DL': ('Digital libraries', 'library, archives, metadata, documents'),
 'cs.DM': ('Discrete mathematics', 'discrete, combinatorics, graph, coloring'),
 'cs.DS': ('Data structures and algorithms', 'algorithm, sorting, data structures, approximation'),
 'cs.FL': ('Formal languages and automata theory', 'automata, grammar, formal language, regular'),
 'cs.GT': ('Computer science and game theory', 'game, equilibrium, auction, mechanism'),
 'cs.HC': ('Human computer interaction', 'interaction, interface, usability, users'),
 'cs.IR': ('Information retrieval', 'retrieval, search, relevance, ranking'),
 'cs.IT': ('Information theory', 'information theory, channel, coding, capacity'),
 'cs.LG': ('Machine learning', 'learning, classification, regression, training'),
 'cs.LO': ('Logic in computer science', 'logic, verification, proof, satisfiability'),
 'cs.MA': ('Multiagent systems', 'agents, coordination, cooperation, multiagent'),
 'cs.MM': ('Multimedia', 'multimedia, video, audio, media'),
 'cs.MS': ('Mathematical software', 'software, numerical, symbolic, computation'),
 'cs.NA': ('Numerical analysis', 'numerical, convergence, discretization, approximation'),
 'cs.NE': ('Neural and evolutionary computing', 'neural, evolutionary, genetic, optimization'),
 'cs.NI': ('Networking and internet architecture', 'network, routing, internet, wireless'),
 'cs.PF': ('Computer performance', 'performance, queueing, latency, throughput'),
 'cs.PL': ('Programming languages', 'programming, compiler, types, semantics'),
 'cs.RO': ('Robotics', 'robot, navigation, motion, control'),
 'cs.SC': ('Symbolic computation', 'symbolic, algebra, polynomial, computation'),
 'cs.SE': ('Software engineering', 'software, testing, debugging, requirements'),
 'cs.SI': ('Social and information networks', 'social network, community, diffusion, graph'),
 'cs.SY': ('Systems and control', 'control, stability, dynamical, feedback'),
 'math.CO': ('Combinatorics', 'combinatorics, enumeration, graph, permutation'),
 'math.IT': ('Information theory', 'information theory, channel, coding, capacity'),
 'math.LO': ('Mathematical logic', 'logic, theorem, decidability, models'),
 'math.NA': ('Numerical analysis', 'numerical, convergence, discretization, approximation'),
 'math.NT': ('Number theory', 'number theory, primes, arithmetic, diophantine'),
 'math.OC': ('Optimization and control', 'optimization, control, convex, constraints'),
 'math.PR': ('Probability', 'probability, stochastic, random, martingale'),
 'math.ST': ('Statistics theory', 'statistics, estimator, inference, asymptotic'),
 'nlin.AO': ('Adaptation and self organizing systems', 'adaptation, self organization, emergence, dynamics'),
 'physics.data-an': ('Data analysis statistics and probability', 'data analysis, statistics, measurement, uncertainty'),
 'physics.soc-ph': ('Physics and society', 'social, population, collective, dynamics'),
 'q-bio.NC': ('Neurons and cognition', 'neuron, brain, cognition, neural'),
 'q-bio.QM': ('Quantitative methods in biology', 'biology, quantitative, biological, inference'),
 'quant-ph': ('Quantum physics', 'quantum, entanglement, qubit, measurement'),
 'stat.AP': ('Statistics applications', 'applied statistics, data, estimation, modeling'),
 'stat.ME': ('Statistics methodology', 'statistical methods, inference, estimation, sampling'),
 'stat.ML': ('Statistical machine learning', 'learning, classification, regression, statistical'),
 'stat.TH': ('Statistics theory', 'statistics, estimator, inference, asymptotic'),
}
CLASSES = list(AAPD_ARXIV_CATEGORIES)
assert set(CATALOG) == set(CLASSES) and len(CLASSES) == 54
DESCRIPTIONS = [CATALOG[c][0] for c in CLASSES]
SEED_TOPICS = [[w.strip() for w in CATALOG[c][1].split(',')] for c in CLASSES]
