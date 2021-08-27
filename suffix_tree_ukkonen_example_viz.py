import string
from graphviz import Digraph
from enum import Enum


class NodeType(Enum):
    ROOT = 1        # root
    INODE = 2       # internal node
    LEAF = 3        # leaf


def visualize_tree(root, txt):
    """
    visualize the suffix tree.
    """
    txt = txt + '#'

    if root.get_children_size() == 0:
        print("<empty>")
        return

    def _print_node(node, level: int):
        """
        recursively print the suffix tree.
        :param node: printing will start from the `node`
        :param level: the level of the tree.
        :return:
        """

        # all transitions at this node (each character may have a transition/edge).
        for c in (string.printable + "\0"):
            # if the the current node has the transition 'c'.
            if c in node.get_all_children():
                child_node = node.get_child(c)
                edge_label = txt[child_node.get_start_index(): child_node.get_end_index()+1]
                if child_node.get_children_size() > 0:
                    print("  " * level, edge_label, " (" + str(child_node.get_total_len()) + ")")
                    # print the internal edge label
                    _print_node(child_node, level + 1)
                else:
                    print("  " * (level - 1), "--", edge_label)

    # print the suffix tree starting from the root.
    print("\n\n------ TREE VISUALIZATION -------")
    print("ROOT")
    _print_node(root, 1)
    print("------ END OF TREE VISUALIZATION -------\n\n")


def visualize_graphviz(tree):
    """
    visualize the suffix tree using the recursion.
    (use the unique identifier of each node. The identifier is the memory address).
    """
    if len(tree.root.get_all_children()) == 0:
        print("<empty>")
        return

    def _print_node(node, level: int):
        if node == tree.root:
            dot.node(str(id(node)), "R")
        else:
            dot.node(str(id(node)), " ")

        # we have ensure that every internal node has
        if node.get_node_type() != NodeType.LEAF:
            dot.edge(str(id(node)), str(id(node.suffix_link)), color="red", style="dashed")

        # all transitions at this node (each character may have a transition/edge).
        for c in (string.printable + '\0'):
            # if the the current node has the transition 'c'.
            if c in node.get_all_children():
                child_node = node.get_child(c)
                edge_label = tree.txt[child_node.get_start_index():child_node.get_end_index()+1]

                # Base Case: if the child node of the current node is a leaf.
                if len(child_node.get_all_children()) == 0:
                    dot.node(str(id(child_node)), str(child_node.get_leaf_id()))
                # Recursion : if the child node of the current can go deeper.
                else:
                    _print_node(child_node, level + 1)  # add '2' to make recursion noticeable....
                dot.edge(str(id(node)), str(id(child_node)), label=edge_label)

    dot = Digraph(comment='Suffix Tree')
    _print_node(tree.root, 0)

    dot.view()
