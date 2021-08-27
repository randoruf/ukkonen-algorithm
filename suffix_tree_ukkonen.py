from __future__ import annotations
from enum import Enum
from typing import Optional
import string
# from suffix_tree_ukkonen_example_viz import visualize_tree, visualize_graphviz


class Path:
    def __init__(self, path_start: int = -1, path_end: int = -1):
        # assert path_end >= path_start
        self._path_start = path_start
        self._path_end = path_end

    def get_path_start(self):
        # assert 0 <= self._path_start <= self._path_end
        return self._path_start

    def get_path_end(self):
        # assert 0 <= self._path_start <= self._path_end
        return self._path_end

    def set_path_start(self, path_start: int):
        # assert path_start >= 0
        self._path_start = path_start

    def set_path_end(self, path_end: int):
        # assert path_end >= 0
        self._path_end = path_end

    def __len__(self):
        if self._path_start == -1 and self._path_end == -1:
            return 0
        else:
            return self._path_end - self._path_start + 1

    def clear(self):
        self._path_start = -1
        self._path_end = -1

    def print_path(self, txt: str):
        if self._path_start == -1 and self._path_end == -1:
            print("Path: <empty>")
        else:
            # print(self._path_start)
            # print(self._path_end)
            print("Path: ", txt[self._path_start:self._path_end+1])


class NodeType(Enum):
    ROOT = 1        # root
    INODE = 2       # internal node
    LEAF = 3        # leaf


class Node:
    global_root = None
    global_end = None

    def __init__(self, node_type: NodeType, start_index: int = -1, end_index: int = -1):
        if node_type != NodeType.ROOT:
            assert self.global_root is not None, "please initialize the global root...."
        # the required attributes for a node
        self._start_index = start_index
        self._end_index = end_index
        self._children = {}
        # other attributes
        self._node_type = node_type
        # the leaf id
        self._leaf_id = -1
        # total path length until this node
        self._total_len = -1
        # parent of the node
        self._parent = None

        # initialize the suffix link
        if node_type == NodeType.ROOT:
            # if this node will be a root, suffix link to itself
            self.suffix_link = self
        else:
            # every node will go back to root for now...
            self.suffix_link = self.global_root

    def set_parent_node(self, pn: Node):
        self._parent = pn

    def get_parent_node(self):
        assert self._parent is not None
        return self._parent

    def get_total_len(self):
        if self._node_type == NodeType.ROOT:
            return 0
        else:
            # assert self._total_len >= 0, "Only internal nodes can have the total length...."
            return self._total_len

    def set_total_len(self, total_len: int):
        # assert self._node_type == NodeType.INODE
        # assert total_len >= 0
        self._total_len = total_len

    def get_node_type(self):
        return self._node_type

    def set_leaf_id(self, j: int):
        # assert self._node_type == NodeType.LEAF, "only leaf has leaf id to express a suffix..."
        # assert j >= 0, "the suffix should start from 0 to n-1"
        # the j th suffixes in the text string... .
        self._leaf_id = j

    def get_leaf_id(self) -> int:
        # assert self._node_type == NodeType.LEAF, "only leaf has leaf id to express a suffix..."
        # assert self._leaf_id >= 0, "you should initialize the leaf if of a leaf...."
        return self._leaf_id

    def set_start_index(self, start_index: int):
        # assert self._node_type != NodeType.ROOT, "the root does not have incoming edges...."
        # assert start_index >= 0, "the starting index should be integers in  [0, n)"
        self._start_index = start_index

    def get_start_index(self) -> int:
        # assert self._node_type != NodeType.ROOT, "the root does not have incoming edges...."
        # assert self._start_index >= 0, "you must initialise the starting index of edge....."
        return self._start_index

    def set_end_index(self, end_index: int):
        # root does not have edge, and leaf has a global end....
        # assert self._node_type == NodeType.INODE, "only internal nodes have ending index of edge...."
        # assert end_index >= 0 and self._start_index <= end_index, "the ending index should be integers in  [0, n)"
        self._end_index = end_index

    def get_end_index(self) -> int:
        if self._node_type == NodeType.LEAF:
            # assert self.global_end is not None
            return self.global_end
        elif self._node_type == NodeType.INODE:
            # assert self._end_index >= 0, "you must initialise the ending index of edge....."
            return self._end_index
        else:
            raise AssertionError("the node should have a type....")

    def add_child_node(self, child_edge_start_char: str, child_node: Node):
        # assert self._node_type != NodeType.LEAF, "leaf should not have any outgoing edges...."
        # assert child_node._start_index >= 0, "you should initialize the first edge of the child..."
        # add the child to the set of outgoing edges from this node.
        self._children[child_edge_start_char] = child_node

    def __len__(self):
        if self._node_type == NodeType.LEAF:
            # assert self._start_index >= 0, "remember to initialize the starting index of the leaf..."
            # assert self.global_end is not None, "remember to give the text as a global end"
            # assert self.global_end >= self._start_index, "forget to update the global end ? Rule 1? "
            # to implement an online ukkonen, the global_end has to be a variable that is increased in each iteration.
            return self.global_end - self._start_index + 1

        elif self._node_type == NodeType.ROOT:
            return 0

        else:
            # assert self._start_index >= 0, "remember to initialize the starting index of the internal node..."
            # assert self._end_index >= 0, "remember to initialize the ending index of the internal node..."
            # assert self._start_index <= self._end_index
            return self._end_index - self._start_index + 1

    def get_child(self, child_edge_start_char: str) -> Node:
        # assert len(child_edge_start_char) == 1
        # assert child_edge_start_char in self._children
        return self._children[child_edge_start_char]

    def has_child(self, child_edge_start_char: str) -> bool:
        return child_edge_start_char in self._children

    def get_all_children(self):
        return self._children

    def get_children_size(self) -> int:
        return len(self._children)

    def set_suffix_link(self, v: Node):
        self.suffix_link = v


class SuffixTree:
    def __init__(self, txt: str):
        assert len(txt) > 0, "non-empty string...."

        # to build a explicit suffix tree....
        self.txt = txt + '\0'

        # create the root for the suffix tree
        self.root = Node(NodeType.ROOT)
        self.root.set_parent_node(self.root)    # note that root's parent is also itself.
        # initialize some variables....
        self.last_j = -1
        Node.global_root = self.root            # all nodes should know which is the root.

        # an internal node that is waiting for the suffix link connection ....
        self.pending_suffix_link: Optional[Node] = None

        # a list of leaves
        self.leaves_list = []

        # ukkonen construction.....
        self.ukkonen()
        # visualize_graphviz(self)

    def ukkonen(self):
        # ----------------------------------------------------------------
        # Base Case : O(1)
        full_suffix_leaf = Node(NodeType.LEAF)
        full_suffix_leaf.set_leaf_id(0)
        self.leaves_list.append(full_suffix_leaf)
        # start and end of the full suffix...
        full_suffix_leaf.set_start_index(0)
        Node.global_end = 0

        # add the leaf to the root (and set leaf's parent as root)
        self.root.add_child_node(self.txt[0], full_suffix_leaf)
        full_suffix_leaf.set_parent_node(self.root)

        # ----------------------------------------------------------------
        # Initialise variables, active node, remainder, last_j and global_end
        active_node = self.root
        remainder = Path()
        self.last_j = 0

        for i in range(1, len(self.txt)):
            Node.global_end += 1
            for j in range(self.last_j+1, i+1):
                # print(i)
                # print("-", j, i)
                # print("-", self.txt[j:i])
                # print("-", self.txt[i])
                # print("....................")
                # ----------------------------------------------------------------
                # update the active node and remainder by traversing down the tree....
                if j == i:
                    # insert a new leaf to the root....
                    active_node = self.root
                    remainder.clear()
                else:
                    # active_node, remainder = self.traverse_down(self.root, Path(j, i-1))
                    active_node, remainder = self._traverse_down(active_node, remainder)

                # ----------------------------------------------------------------
                # apply the appropriate extension txt[i]
                if self._make_extension(j, i, active_node, remainder) == 3:
                    # print("RULE 3")
                    # 1. resolve any pending suffix links from the previous extension
                    if self.pending_suffix_link is not None:
                        self.pending_suffix_link.set_suffix_link(active_node)
                        self.pending_suffix_link = None

                    # 2. prepare to move to next phase (lecture note book, pp57 Rule 3 extensions in Ukkonen's)
                    #    and update the remainder (but don't follow the suffix link)....
                    if len(remainder) > 0:
                        remainder.set_path_end(remainder.get_path_end() + 1)
                    else:
                        remainder.set_path_start(i)
                        remainder.set_path_end(i)
                    # remainder.print_path(self.txt)
                    # 3. terminate this phase early i.e. showstopper
                    break
                else:
                    # print("RULE 2")
                    # 1. resolve any pending suffix links from the previous extension
                    #    (it has been resolved in the make_extension function....)

                    # 2. follow the suffix link and update the active node and remainder...
                    if active_node == self.root:
                        # if u = r, then v = r, remove the first character from the remainder...
                        if len(remainder) <= 1:
                            remainder.clear()
                        if len(remainder) > 1:
                            remainder.set_path_start(remainder.get_path_start() + 1)
                    else:
                        # if u != r, u must be an internal node.
                        # follow the suffix link and keep the remainder...
                        active_node = active_node.suffix_link

                    # print(j+1, i+1)
                    # remainder.print_path(self.txt)

                    # 3. move to next extension
                    #   (i.e. traversing suffix links (lecture note book, pp49 General extension procedure)
                # print("---------------------------------------------------------------------------------------------")

        # visualize_tree(self.root, self.txt)

    def _traverse_down(self, acn: Node, rem: Path):
        """
        This function essentially update the active node and remainder by the skip-count trick....
        :param acn: active node
        :param rem: remainder
        :return:
        """

        # if the given path is empty....
        if len(rem) == 0:
            rem.clear()
            return acn, rem

        # n is in fact the active node
        remainder_len = len(rem)
        start_idx = rem.get_path_start()
        n_parent = acn
        n = acn.get_child(self.txt[start_idx])

        while remainder_len > 0:
            if remainder_len < len(n):
                break
            if remainder_len == len(n):
                rem.clear()
                return n, rem
            else:
                start_idx += len(n)
                remainder_len -= len(n)
                n_parent = n
                n = n.get_child(self.txt[start_idx])

        # update the remainder....
        rem.set_path_start(n.get_start_index())
        rem.set_path_end(n.get_start_index() + remainder_len - 1)
        return n_parent, rem

    def _make_extension(self, j: int, i: int, active_node: Node, remainder: Path):
        if len(remainder) == 0:
            if not active_node.has_child(self.txt[i]):
                # print("RULE 2 - only add a leaf")
                new_leaf = Node(NodeType.LEAF, i)
                new_leaf.set_leaf_id(j)
                # append the new leaf to the current active node and set new_leaf's parent as current active node
                active_node.add_child_node(self.txt[i], new_leaf)
                new_leaf.set_parent_node(active_node)
                # add leaves to a static list, so easy to manage all leaves....
                self.leaves_list.append(new_leaf)
                self.last_j += 1

                # the previous internal node that waiting for suffix link should connect the current active node.
                if self.pending_suffix_link is not None:
                    self.pending_suffix_link.set_suffix_link(active_node)
                    self.pending_suffix_link = None

                return 2
            else:

                # print("RULE 3")
                return 3
        else:
            if self.txt[i] == self.txt[remainder.get_path_end()+1]:
                # print("RULE 3")
                return 3

            else:
                # print("RULE 2 - add a leaf and an internal node")
                new_internal_node = Node(NodeType.INODE, remainder.get_path_start(), remainder.get_path_end())

                # the node under the new internal node (it starting edge index and parent change!!)
                active_node.get_child(self.txt[remainder.get_path_start()]).set_start_index(remainder.get_path_end()+1)
                active_node.get_child(self.txt[remainder.get_path_start()]).set_parent_node(new_internal_node)

                # the new internal should take current active node's child, and internal node's parent as active node..
                new_internal_node.add_child_node(self.txt[remainder.get_path_end()+1],
                                                 active_node.get_child(self.txt[remainder.get_path_start()]))
                new_internal_node.set_parent_node(active_node)

                # only internal node has the total path length..... ()
                new_internal_node.set_total_len(active_node.get_total_len() + len(remainder))

                # create a new leaf because of rule 2
                new_leaf = Node(NodeType.LEAF, i)
                new_leaf.set_leaf_id(j)
                # (append the new leaf to the new internal node, because of rule 2, and don't forget leaf's parent)....
                new_internal_node.add_child_node(self.txt[i], new_leaf)
                new_leaf.set_parent_node(new_internal_node)
                # add leaves to a static list, so easy to manage all leaves....
                self.leaves_list.append(new_leaf)

                # update active node's child, which should be the new internal node....
                active_node.add_child_node(self.txt[remainder.get_path_start()], new_internal_node)

                # since rule 2 is applied, the last_j indicate the last rule 2 extension, so increase it.
                self.last_j += 1

                # the previous internal node that waiting for suffix link should connect this new internal node.
                if self.pending_suffix_link is not None:
                    self.pending_suffix_link.set_suffix_link(new_internal_node)
                    self.pending_suffix_link = None

                # now this new internal node is waiting for suffix link
                self.pending_suffix_link = new_internal_node

                return 2

    def search_tree_traversal(self, q: str):
        """
        Assume a tuple (node, k).
        node is the current node,
        k is the position of the edge label where comparison stops (substring)

        Traverse the tree following the path given by the query string q.
            - Case 1: If we fall off the tree, that means not a substring of suffix.
            - Case 2a: If we stop at the middle of the edge, that means a substring.
            - Case 2b: If we finish at a node, that means a true suffix.
        :param q: the query string
        :return:
        """
        curr_node = self.root
        j = 0
        while j < len(q):
            c = q[j]
            # Case 1: fell off the tree (fell off at node)
            if not curr_node.has_child(c):
                return None
            else:
                child = curr_node.get_child(c)
                edge_label = self.txt[child.get_start_index():child.get_end_index()+1]
                k = j + 1
                while k < len(q) and k - j < len(edge_label) and q[k] == edge_label[k - j]:
                    k += 1
                # Next iteration: exhausted edge (shift to the next node and go further)
                if k - j == len(edge_label):
                    curr_node = child
                    j = k
                # can't dive deeper into the tree
                # Case 2a : exhausted query string in middle of edge (substring)
                elif k == len(q):
                    curr_node = child
                    if len(curr_node.get_all_children()) == 0:
                        # print("substring")
                        # print(curr_node.leaf_id)
                        pass
                    return curr_node
                # Case 1: fell off in the middle of edge (since q[k] != edge_label[k]) breaks..
                else:
                    return None

        # Case 2b: exhausted query string at a node
        #   (just next to a node, but don't fall off the tree).
        return curr_node

    @staticmethod
    def dfs(n: Node):
        """
        use the dfs for tree traversing from the node n

        NOTE: the 'visited property' is not important in a tree (compared to graph).
              (why? we can't go back. so don't need to mark the node as visited).
        :param n: a node in the tree
        :return:
        """
        # the full alphabet set....
        alphabet_set = string.printable[::-1] + '\0'

        # the stack for DFS
        stack = [n]
        ans = []
        while len(stack) > 0:
            v = stack.pop()
            if len(v.get_all_children()) == 0:
                # at a leaf node (return the answer)
                ans.append(v.get_leaf_id())
            else:
                for c in alphabet_set:
                    if v.has_child(c):
                        stack.append(v.get_child(c))

        # ans.sort()  # sort is recommended to be done outside.
        return ans

    def search_substring(self, q: str):
        # follow the path of the query string q, to the nearest next node.
        node = self.search_tree_traversal(q)
        if node is not None:
            # traversal all leaves in a tree (https://en.wikipedia.org/wiki/Tree_traversal)
            return self.dfs(node)
        else:
            return []

####################################################################################


def check_eq(actual, expected):
    if actual != expected:
        print('actual: %s, expected: %s' % (actual, expected))
    else:
        print('Pass')


def match(txt: str, pat: str):
    s = SuffixTree(txt)
    ans = s.search_substring(pat)
    ans.sort()
    return ans


if __name__ == "__main__":
    print()

    # t1 = "abacabad"
    # s1 = SuffixTree(t1)
    # visualize_tree(s1.root, t1)

    t2 = "aabbabaa"
    s2 = SuffixTree(t2)
    # visualize_tree(s2.root, t2)

    # t3 = "abdyabxdcyabcdzacdabcaxcd"
    # s3 = SuffixTree(t3)

    check_eq(match("A", "bbbbb"), [])
    check_eq(match("A", "A"), [0])
    check_eq(match("ABC ABCDAB ABCDABCDABDE", "ABCDABD"), [15])
    check_eq(match("ABC#ABCDAB#ABCDABCDABDE", "AB"), [0, 4, 8, 11, 15, 19])
    check_eq(match("ABC ABCDAB ABCDABCDABDE", "B"), [1, 5, 9, 12, 16, 20])
    check_eq(match("AAAAA", "A"), [0, 1, 2, 3, 4])
    check_eq(match("AAAAA", "AA"), [0, 1, 2, 3])
    check_eq(match("AAAAA", "AAAA"), [0, 1])
    check_eq(match("AAAAA", "AAAAA"), [0])
    check_eq(match("AABAABA", "AABA"), [0, 3])
