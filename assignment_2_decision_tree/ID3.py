from collections import Counter, defaultdict
from graphviz import Digraph
import math


class ID3DecisionTreeClassifier:
    def __init__(self, minSamplesLeaf=1, minSamplesSplit=2):

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment="The Decision Tree")

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {
            "id": self.__nodeCounter,
            "value": None,
            "label": None,
            "attribute": None,
            "entropy": None,
            "samples": None,
            "classCounts": None,
            "nodes": None,
        }

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ""
        for k in node:
            if (node[k] != None) and (k != "nodes"):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node["id"]), label=nodeString)
        if parentid != -1:
            self.__dot.edge(str(parentid), str(node["id"]))
            nodeString += "\n" + str(parentid) + " -> " + str(node["id"])

        print(nodeString)

        return

    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self):

        # Change this to make some more sense
        return None

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        # Create a (root) node Root for the tree
        root = self.new_ID3_node()
        root.update({"value": "hello world"})
        self.add_node_to_graph(root)

        # Count classes
        classes_with_count = self.count_class_members(target)
        most_common_class = self.find_most_common_class(classes_with_count)

        # If all samples belong to one class return the single node tree root with label <class_name>
        if len(classes_with_count.keys()) == 1:
            root.update({"value": classes_with_count.keys()[0]})
            return

        # If Attributes is empty, then
        # Return the single node tree Root, with label = most common class value in Samples.
        if not attributes:
            root.update({"value": most_common_class})
            return
        else:
            
        return root

    def predict(self, data, tree):
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted

    def entropy(self, total_value_count, classes):
        entropy = 0
        for c in classes.keys():
            p_x = classes[c] / total_value_count
            entropy += -p_x * math.log(p_x, 2)
        return entropy

    def count_class_members(self, target):
        classes_with_count = defaultdict(int)
        for t in target:
            classes_with_count[t] += 1
        return classes_with_count

    def find_most_common_class(self, classes):
        most_common_class = max(classes, key=lambda c: classes[c])
        return most_common_class