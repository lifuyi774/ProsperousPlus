/*
 * BWPTree.java
 *
 * Created on 27. Juni 2006, 21:45
 *
 * To change this template, choose Tools | Options and locate the template under
 * the Source Creation and Management node. Right-click the template and choose
 * Open. You can then make changes to the template in the Source Editor.
 */

package weka.classifiers.trees;

import weka.classifiers.trees.bwptree.Node;
import weka.core.*;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import java.io.*;
import java.util.*;
import java.lang.Math;

/**
 *
 * @author Ceilican
 */
public class BWPTree extends Classifier implements Serializable {
    
    /** Stores the dataset */
    private Instances dataset;
    
    /** The Root of the Decision Tree */
    private Node treeRoot;
    
    /** Creates a new instance of BWPTree */
    public BWPTree() {
    }    
     
    /**
    * Returns a string describing classifier
    * @return a description suitable for
    * displaying in the explorer/experimenter gui
    */
    public String globalInfo() {
        return "Class for building and using a simple Decision Tree classifier.";	    
    }

    /**
    * Generates the classifier.
    *
    * @param instances set of instances serving as training data 
    * @exception Exception if the classifier has not been generated successfully
    */
    public void buildClassifier(Instances instances) throws Exception {
        dataset = instances;
        treeRoot = new Node(dataset,0.85);
        System.out.print(treeRoot.toString());
    }

    /**
    * Classifies a given instance.
    *
    * @param instance the instance to be classified
    * @return index of the predicted class
    */
    public double classifyInstance(Instance instance) {
        Node currentNode = treeRoot;
        while (!currentNode.isLeaf()) {
            String attValue = instance.stringValue(currentNode.getAttIndex());
            int childPosition = instance.attribute(currentNode.getAttIndex()).indexOfValue(attValue);
            currentNode = currentNode.getChild(childPosition);
        }
        //System.out.println("(classifyInstance) instance classified as: " + currentNode.getLeafClassValueIndex());
        return (double) currentNode.getLeafClassValueIndex();
    }
 
  /**
   * Returns a description of the classifier.
   *
   * @return a description of the classifier as a string.
   */
  public String toString() {
     return "BWPTree is a Decision Tree Classifier based on ID3 with pre-pruning when " +
            "the entropy level falls below an entropyThreshold equal to: " + treeRoot.getEntropyThreshold();
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
    
    try {
      //String argvAlt [] ={"-t","weathernominal.arff"};
      String argvAlt [] ={"-t","spambase_entrodiscretized.arff"};
      System.out.println(Evaluation.evaluateModel(new BWPTree(), argvAlt));
      //System.out.println(Evaluation.evaluateModel(new BWPTree(), argv));
    } catch (Exception e) {
      e.printStackTrace();  
      System.err.println(e.getMessage());
    }
    
//    try {  
//        //BufferedReader inputReader = new BufferedReader(new FileReader("spambase_discretized_reduced.arff"));
//        //BufferedReader inputReader = new BufferedReader(new FileReader("spambase_discretized.arff"));
//        BufferedReader inputReader = new BufferedReader(new FileReader("weathernominal.arff"));
//        Instances dataset = new Instances(inputReader);
//        dataset.setClassIndex(dataset.numAttributes()-1);
//        BWPTree tree = new BWPTree();
//        tree.buildClassifier(dataset);
//        tree.classifyInstance(dataset.instance(0));
//    } catch (Exception e) {
//        System.out.println("main " + e);
//    }
      
  }
  
}
