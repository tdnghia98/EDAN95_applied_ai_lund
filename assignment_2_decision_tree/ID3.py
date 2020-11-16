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
    def new_ID3_node(self, name=None):
        node = {
            "id": self.__nodeCounter,
            "value": None,
            "label": None,
            "attribute": None,
            "entropy": None,
            "samples": None,
            "classCounts": None,
            "nodes": {},
            "note": None,
        }
        print("New node created %s" % name)
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

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(
        self, data, target, attributes, classes, value=None, remaining_attributes=None
    ):

        # Create a (root) node Root for the tree
        root = self.new_ID3_node("root")

        # Count classes
        classes_with_count = self.count_class_members(target)
        most_common_class = self.find_most_common_class(classes_with_count)
        current_entropy = self.entropy(data, classes_with_count)
        # If all samples belong to one class return the single node tree root with label <class_name>
        if len(classes_with_count.keys()) == 1:
            root.update(
                {
                    "label": most_common_class,
                    "value": "-" if not value else value,
                    "samples": list(classes_with_count.values())[0],
                    "entropy": current_entropy,
                    "classCounts": classes_with_count,
                    "note": "only one class",
                }
            )
            self.add_node_to_graph(root)
            return root

        # If Attributes is empty, then
        # Return the single node tree Root, with label = most common class value in Samples.
        if not remaining_attributes:
            root.update(
                {
                    "label": most_common_class,
                    "value": "-" if not value else value,
                    "samples": len(data),
                    "entropy": current_entropy,
                    "classCounts": classes_with_count,
                    "note": "no remaining att",
                }
            )
            self.add_node_to_graph(root)
            return root

        # Let A be the attribute a in Attributes that generates the maximum information gain
        # when the tree is split based on a.
        (
            _,
            max_split_attribute,
            max_subsets,
            max_targets,
        ) = self.get_max_information_gain(
            data, attributes, remaining_attributes, target, current_entropy
        )
        # Set A as the target_attribute of Root
        root.update(
            {
                "attribute": max_split_attribute,
                "value": "-" if not value else value,
                "samples": len(data),
                "entropy": current_entropy,
                "classCounts": classes_with_count,
                "note": "root node",
            }
        )

        #   Let Samples(v) be the subset of samples that have the value v for A.
        #               If Samples(v) is empty, then
        #                   Below this new branch add a leaf node with label
        #                         = most common class value in Samples.
        #               else
        #                   Below this new branch add the subtree ID3 (Samples(v), A, Attributes/{A})

        for attribute_value in attributes[max_split_attribute]:
            samples = max_subsets[attribute_value]
            sample_target = max_targets[attribute_value]
            node = None
            if not samples:
                node = self.new_ID3_node("child")
                node.update(
                    {
                        "value": "-" if not value else value,
                        "label": most_common_class,
                        "samples": 0,
                        "classCounts": 0,
                        "note": "No sample",
                    }
                )
            else:
                remaining_attributes = remaining_attributes.copy()
                remaining_attributes.pop(max_split_attribute, None)
                node = self.fit(
                    samples,
                    sample_target,
                    attributes,
                    classes,
                    value=attribute_value,
                    remaining_attributes=remaining_attributes,
                )

            root["nodes"][attribute_value] = node
            self.add_node_to_graph(node, root["id"])
        self.add_node_to_graph(root)

        return root

    def predict(self, data, tree):
        predicted = list()

        for element in data:
            label = self.traverse_tree(element, tree)
            predicted.append(label)

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted

    def traverse_tree(self, element, tree):
        if len(tree["nodes"]) == 0:
            return tree["label"]
        else:
            for child in tree["nodes"].values():
                if child["value"] in element:
                    return self.traverse_tree(element, child)

    def entropy(self, data, classes_with_count):
        entropy = 0
        for c in classes_with_count.keys():
            p_x = classes_with_count[c] / len(data)
            entropy += -p_x * math.log(p_x, 2)
        return entropy

    def count_class_members(self, target):
        classes_with_count = defaultdict(int)
        for t in target:
            classes_with_count[t] += 1
        return classes_with_count

    def find_most_common_class(self, classes_with_count):
        most_common_class = max(classes_with_count, key=lambda c: classes_with_count[c])
        return most_common_class

    def get_max_information_gain(
        self, data, attributes, remaining_attributes, target, current_entropy
    ):
        max_information_gain = float("-inf")
        max_split_attribute = None
        max_subsets, max_targets = None, None
        for attribute in remaining_attributes:
            attribute_idx = list(attributes.keys()).index(attribute)
            (
                attribute_entropy,
                subsets_data,
                subsets_target,
            ) = self.get_attribute_entropy(data, target, attribute_idx)
            information_gain = current_entropy - attribute_entropy
            if information_gain > max_information_gain:
                max_split_attribute = attribute
                max_information_gain = information_gain
                max_subsets = subsets_data
                max_targets = subsets_target
        return max_information_gain, max_split_attribute, max_subsets, max_targets

    def get_attribute_entropy(self, data, target, split_attribute_idx):
        subsets_data = defaultdict(list)
        subsets_target = defaultdict(list)
        attribute_entropy = 0
        # Generate subsets of the attribute (Sv of A)
        for idx, row in enumerate(data):
            attribute_value = row[split_attribute_idx]
            subsets_data[attribute_value].append(row)
            subsets_target[attribute_value].append(target[idx])

        for attribute_value in subsets_data.keys():
            subset = subsets_data[attribute_value]
            class_count = self.count_class_members(subsets_target[attribute_value])
            attribute_entropy += (
                len(subset) / len(data) * self.entropy(subset, class_count)
            )

        return attribute_entropy, subsets_data, subsets_target
