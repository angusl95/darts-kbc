import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("e_s", fillcolor='darkseagreen2')
  g.node("e_r", fillcolor='darkseagreen2')
  g.node("e_o", fillcolor='darkseagreen2')
  #assert len(genotype) % 2 == 0
  #steps = len(genotype) // 2
  steps = len(genotype)

  for i in range(steps+1):
    g.node(str(i), fillcolor='lightblue')

  g.edge("e_s", "0", label="interleave", fillcolor="gray")
  g.edge("e_r", "0", label="interleave", fillcolor="gray")

  for i in range(steps):
      op, _ = genotype[i]
      g.edge(str(i), str(i+1), label=op, fillcolor="gray")

  g.node("proj", fillcolor='palegoldenrod')
  g.node("f", fillcolor='palegoldenrod')

  g.edge(str(steps), "proj", label="linear", fillcolor="gray")
  
  #for i in range(steps):
  #  g.edge(str(i), "c_{i}", fillcolor="gray")
  g.edge("proj", "f", label="dot product", fillcolor="gray")
  g.edge("e_o", "f", label="dot product", fillcolor="gray" )
  g.render(filename, view=True)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot(genotype.normal, "normal")
  #plot(genotype.reduce, "reduction")

