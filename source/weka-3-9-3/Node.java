/*
 * Node.java
 *
 * Created on 2. Juli 2006, 17:56
 *
 * To change this template, choose Tools | Options and locate the template under
 * the Source Creation and Management node. Right-click the template and choose
 * Open. You can then make changes to the template in the Source Editor.
 */

package weka.classifiers.trees.bwptree;

import weka.core.*;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import java.io.*;
import java.util.*;
import java.lang.Math;

/**
 *  Node is the class for nodes in the Decision Tree
 *
 * @author Ceilican
 */
public class Node implements Serializable {
    /** Points to the entire dataset that is being learned */
    private Instances dataset;
    
    /** Holds the position of the attribute associated with the node. (-1) if the node is a leaf. */
    private int attIndex = -1;
        
    /** Holds the positions of the instances of the dataset which belong to the partition associated with the node */
    private Vector nodePartition;
    
    /** Holds the children of this node. The children are in the order of the attribute values of this node's attribute. */
    private Node [] children = null;
    
    /** Holds the entropy level below which the growth of the tree should be stopped*/
    private double entropyThreshold;
    
    /** Holds the index of the value of the class, if the node is a leaf. (-1) if it is not a leaf. */
    private int leafClassValueIndex = -1;
    
    /** Each position of the vector holds whether the corresponding attribute
     *  is associated with an antecessor node of this node. This saves a lot of
     *  computation, because these antecessor attributes don't need to be 
     *  considered as candidates for this node */
    private boolean [] isAntecessorAttribute;
    
    /*
     * Constructor to construct a root.
     */
    public Node(Instances dataset, double entropyThreshold) {
        this.dataset = dataset;
        this.entropyThreshold = entropyThreshold;
        nodePartition = new Vector(dataset.numInstances());
        for (int i=0;i<dataset.numInstances();i++) {
            nodePartition.add(i);
        }
        isAntecessorAttribute = new boolean[dataset.numAttributes()];
        isAntecessorAttribute[dataset.classIndex()] = true; //The class attribute should never be considered to be a candidate attribute for a node.
        selectBestAttributeAndGrowChildren();
    }
    
    public Node(Instances dataset, Vector childPartition, boolean [] isAntecessorAttribute, double entropyThreshold) {
        this.entropyThreshold = entropyThreshold;
        this.dataset = dataset;
        nodePartition = childPartition;
        this.isAntecessorAttribute = isAntecessorAttribute;
        selectBestAttributeAndGrowChildren();
    }
    
    public double getEntropyThreshold(){
        return entropyThreshold;
    }
    
    public int getAttIndex() {
        return attIndex;
    }
    
    public int getLeafClassValueIndex() {
        return leafClassValueIndex;
    }
    
    public Node getChild(int childPosition) {
        try {
            Node child = children[childPosition];
            return child;
        } catch (Exception e) {return null;}
    }
    
    public double gain(int attIndex) {
        Vector[] newPartitions = split(attIndex);
        return gain(newPartitions);
    }
    
    private double gain(Vector[] newPartitions){
        double gain = selfEntropy();
        gain -= newPartitionsEntropy(newPartitions);
        return gain;
    }
    
    private double newPartitionsEntropy(Vector[] newPartitions){
        double weightedEntropy = 0;
        for (int i=0; i<newPartitions.length; i++) {
            weightedEntropy += (newPartitions[i].size()/(1.0*nodePartition.size()))*partitionEntropy(newPartitions[i]);
        }        
        return weightedEntropy;
    }
    
    private boolean noAttributesLeft() {
        for (int i=0; i<isAntecessorAttribute.length; i++) { //if any attribute is not in the antecessors array, then there is at least one attributes left. And thus the method return false.
            //System.out.println("noAttributesleft: "+ i+ isAntecessorAttribute[i]);
            if (!isAntecessorAttribute[i]) {
                return false;
            }
        }
        return true;
    }
    
    private int majorityVote(Vector partition) {
        int [] classCount = classCount(partition);
        int majorClassValue = -1;
        int bestCount = -1;
        
        for (int i=0; i<classCount.length;i++) {
            if (classCount[i]>bestCount) {
                majorClassValue = i;
                bestCount = classCount[i];
            }
        }
        
        return majorClassValue;
    }
    
    public int selectBestAttributeAndGrowChildren() {
        if (selfEntropy() <= entropyThreshold || noAttributesLeft()) { //pre-pruning the tree. stoping the growth if the entropy of the node is already smaller than the threshold.
            children = null;
            leafClassValueIndex = majorityVote(nodePartition);
            //System.out.println("leaf class: " + leafClassValueIndex);
            return -1;
        } 
        else {
            double bestMinimumNewEntropy = 1.0;
            int bestAtt = -1;
            Vector [] bestNewPartitions = null;

            for (int i=0; i<dataset.numAttributes(); i++) {
                if (!isAntecessorAttribute[i]) {
                    Vector [] candidateNewPartitions = split(i);
                    double candidateNewPartitionsEntropy = newPartitionsEntropy(candidateNewPartitions);
                    if (candidateNewPartitionsEntropy <= bestMinimumNewEntropy) {
                        bestMinimumNewEntropy = candidateNewPartitionsEntropy;
                        bestAtt = i;
                        bestNewPartitions = candidateNewPartitions;
                    }
                }
            }
            attIndex = bestAtt;
            generateChildren(bestNewPartitions);

            //System.out.println("BestAttribute is "+bestAtt+" with a gain equal to "+gain(bestAtt)+" and number of partitions: "+bestNewPartitions.length);
            return bestAtt;
        } 
    }
    
    private void generateChildren(Vector [] bestNewPartitions){
        boolean [] newIsAntecessorAttribute = isAntecessorAttribute.clone();
        //System.out.println("generateChildren attindex: "+attIndex);
        newIsAntecessorAttribute[attIndex] = true;
        children = new Node[bestNewPartitions.length];
        for (int i=0;i<bestNewPartitions.length;i++) {
            //System.out.println("generate children i: "+i);
            children[i] = new Node(dataset, bestNewPartitions[i], newIsAntecessorAttribute, entropyThreshold);
        }
    }
    
    /*
     * Splits a partition into new partitions corresponding to the 
     * values of the given attribute.
     */
    public Vector[] split(int attIndex) {
        Vector newPartitions [] = new Vector[dataset.attribute(attIndex).numValues()];
        for (int i=0; i<newPartitions.length; i++) {
            //System.out.println(i);
            newPartitions[i] = new Vector();
        }
        
        for (int i=0; i<nodePartition.size(); i++) {
            Instance inst = dataset.instance(((Integer) nodePartition.elementAt(i)).intValue());
            String attValue = inst.stringValue(attIndex);
            int attValueIndex = inst.attribute(attIndex).indexOfValue(attValue);
            //System.out.println(attValueIndex);
            newPartitions[attValueIndex].add(nodePartition.elementAt(i));
            
        }
        
        return newPartitions;
    }
    
    public double selfEntropy(){
        return partitionEntropy(nodePartition);
    }
    
    private int[] classCount(Vector partition) {
        // Inititalizing the array that counts the ocurrences of 
        // the different values of the class attribute in the parition
        int numClassValues = dataset.classAttribute().numValues();
        int classCount[] = new int[numClassValues];
        
        // Counting the ocurrences of the values in the partition
        for (int i=0; i<partition.size(); i++) {
            Instance inst = dataset.instance(((Integer) partition.elementAt(i)).intValue());
            String classValue = inst.stringValue(inst.classIndex());
            int classValueIndex = inst.classAttribute().indexOfValue(classValue);
            //System.out.println(classValueIndex);
            classCount[classValueIndex]++;
        }
        return classCount;
    }
    
    private double partitionEntropy(Vector partition){
        int classCount[] = classCount(partition);
        
        // Computing the entropy
        double entropy = 0.0;
        double normalizedCount [] = new double[classCount.length];
        for (int j=0; j<classCount.length; j++) {
            //System.out.println(j);
            normalizedCount[j] = classCount[j]/(1.0 * partition.size());
            if (classCount[j]!=0) { // this avoids undeterminate results when 0.log(0) is computed
                entropy -= normalizedCount[j]*(Math.log(normalizedCount[j])/Math.log(2));
            }
        }
        
        return entropy;
    }
    
    public boolean isLeaf() {
        if (children!=null) return false;
        else return true;
    }
    
    public String toString() {
        return "The learned tree is:\n"+toString(0);
    }
    private String toString(int tab) {
        String text="";
        if (!isLeaf()) {
            text += dataset.attribute(attIndex).name() + "\n";
            for (int i=0;i<children.length-1;i++) {
                text += tab(tab+1)+"("+dataset.attribute(attIndex).value(i)+")>> "+ children[i].toString(tab+1)+"\n";
            }
            text += tab(tab+1)+"("+dataset.attribute(attIndex).value(children.length-1)+")>> "+ children[children.length-1].toString(tab+1);
        } else {
            text += dataset.classAttribute().value(leafClassValueIndex);
        }
            
        return text;
    }  
    private String tab(int tab) {
        String tabString = "";
        for (int i=0; i<tab;i++) {
            tabString += "   ";
        }
        return tabString;
    }
}
