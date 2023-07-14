package weka.filters.supervised.attribute;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.TreeSet;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.filters.supervised.attribute.Discretize;
import java.io.File;
/**
 * @author Matthias
 *
 */
public class EntroDiscretize extends Filter implements SupervisedFilter {
	
	// gives a list of Lists for the cutpoints of all attributes
	ArrayList<TreeSet<Double>> attributesCutPoints=new ArrayList<TreeSet<Double>>();
	
	public boolean setInputFormat(Instances instanceInfo) throws Exception {
		
		try
		{
		if (instanceInfo.classAttribute().isNominal() &&
				instanceInfo.classAttribute().numValues()>2) 
			throw new UnsupportedClassTypeException("Only two Classes are allowed");
		
	    super.setInputFormat(instanceInfo);
	    
	    attributesCutPoints.clear();
		}
		catch (Exception ex) {
		      System.out.println(ex.toString());
		    }
	    return false;
	  }
	
	
	public boolean input(Instance instance) {

	    if (getInputFormat() == null) {
	      throw new IllegalStateException("No input format");
	    }
	    if (m_NewBatch) {
	      resetQueue();
	      m_NewBatch = false;
	    }
	    
	    if (!attributesCutPoints.isEmpty()) {
	      convertInstance(instance);
	      return true;
	    }

	    bufferInput(instance);
	    return false;
	  }
	
	 public boolean batchFinished() {
		 	
		 	
		    if (getInputFormat() == null) {
		      throw new IllegalStateException("No input instance format defined");
		    }
		    
		    try
		    {
		    	if (attributesCutPoints.size()==0)
			    {
			    	findCutPointsforAllAttributes();
			    	
			    	System.out.println("Calculated CutPoints:"+attributesCutPoints.size());
			    	System.out.println(attributesCutPoints.toString());
			    	
			    	setOutputFormat();
			    	// convert all instances
			    	for(int i = 0; i < getInputFormat().numInstances(); i++) {
						convertInstance(getInputFormat().instance(i));}
			    
			    	
			    }
			    
			    		    
			    flushInput();
			     
			    m_NewBatch = true;
			    
			    return (numPendingOutput() != 0);
		    }
		    catch (Exception ex) {
		      System.out.println("Exception in batchFinished: "+ex.toString());
		      return false;
		    }
		    
		  }

	
	/**
	 * Calculates the cutpoints for all numeric attributes
	 */
	public void findCutPointsforAllAttributes()
	{
		try
		{
			if (getInputFormat()==null) throw new IllegalStateException("No input format");
			
			// make copy of real instances to work with
			Instances actualInstances=new Instances(getInputFormat()); 
			
			//System.out.println(actualInstances.toString());
			
			
			int listIndex=0;
			
			for (int i=0;i<=actualInstances.numAttributes()-1;i++)
			{
				if(actualInstances.attribute(i).isNumeric())
				{
					attributesCutPoints.add(new TreeSet<Double>());
					actualInstances.sort(i);
					/*
					System.out.println(i+"th attribute:");
					for(int a=0;a<actualInstances.numInstances();a++)
					{
						System.out.println(actualInstances.instance(a).value(i));
					}*/
					
					cutPointForOneAttrib(actualInstances,i,listIndex,0,actualInstances.numInstances()-1);
					listIndex++;
				}
			}
		}
		catch (Exception ex) {
		      System.out.println("Exception in findCutPointsforAllAttributes: "+ex.toString());
		      
		    }
		
		
		
	}
		

	/**
	 * @param copy gives the instanceset to work with
	 * @param attrIndex gives the index of the actual attribute
	 * @param index in the ArrayList of cutpoints
	 * @param lowerBound determines the first element in the intervall
	 * @param upperBound determines the last element in the intervall
	 * @return if a new Cut Point was set with respect to the stopping criterium
	 * 
	 */
	
	public void cutPointForOneAttrib(Instances copy,int attrIndex,int listIndex,int lowerBound, int upperBound)
	{
		try
		{
//			 find the best cutpoint
			int cutInstance=lowerBound; // cutInstance belongs to the left intervall
			double minEntropy=1;
			//try all possibilities
			for(int i=lowerBound;i<upperBound;i++)
			{
				// do the boundaries only between classes (Annäherung an andere Vorgehensweise)
				while(copy.instance(i).classValue()==copy.instance(i+1).classValue()&& i<upperBound-1 )i++;
				
				
				// size of left intervall=i-lowerBound +1 ; size of rigth intervall=upperBound-i;
				double sizeLeft=i-lowerBound+1;
				double sizeRight=upperBound-i;
				double actualEntr=sizeLeft/(sizeLeft+sizeRight)*entropy(copy,lowerBound,i)
								+ sizeRight/(sizeLeft+sizeRight)*entropy(copy,i+1,upperBound);
				if (actualEntr<minEntropy) { minEntropy=actualEntr;cutInstance=i;}
				
			}
			
			if (!stoppingCriterium(copy,lowerBound,upperBound,cutInstance,minEntropy)) 
			{
				// if criterium isn't fullfiled , add cutpoint and apply function to resulting intervalls
				attributesCutPoints.get(listIndex).add(copy.instance(cutInstance).value(attrIndex));
				// left intervall
				cutPointForOneAttrib(copy,attrIndex,listIndex,lowerBound,cutInstance);
				// rigth intervall
				cutPointForOneAttrib(copy,attrIndex,listIndex,cutInstance+1,upperBound);
				
			}
		}
		catch (Exception ex) {
		      System.out.println("Exception in cutPointForOneAttrib: "+ex.toString());
		      
		    }
				
		
	}
	
	/**
	 * @param copy gives the already ordered instance set
	 * @param lowerBound gives the index of the first element in the intervall
	 * @param upperBound gives the index of the last element in the intervall
	 * @param cutInstance the last instance in the left intervall
	 * @return true, if criterium is fullfiled 
	 */
	public boolean stoppingCriterium(Instances copy,int lowerBound, int upperBound,int cutInstance,double entropyForCuting)
	{
	
		int numberClassesLeft=numberOfClasses(copy,lowerBound,cutInstance);
		int numberClassesRight=numberOfClasses(copy,cutInstance+1,upperBound);
		int numberClasses=numberOfClasses(copy,lowerBound,upperBound);
		int numberOfInstances=upperBound-lowerBound+1;
		double gain=0;
		double delta=0;
		double entropyForWholeSet=entropy(copy,lowerBound,upperBound);
		
		delta = Utils.log2(Math.pow(3, numberClasses) - 2) - 
	      (((double) numberClasses * entropyForWholeSet) - 
	       (numberClassesLeft * entropy(copy,lowerBound,cutInstance)) - 
	       (numberClassesRight * entropy(copy,cutInstance+1,upperBound)));

		gain=entropyForWholeSet-entropyForCuting;
		
		//System.out.println("gain"+gain);
		//System.out.println("the other"+(Utils.log2(numberOfInstances-1)+delta )/(double) numberOfInstances );
		
		return (gain < (Utils.log2(numberOfInstances-1)+delta )/(double) numberOfInstances );
	}
	
	public int numberOfClasses(Instances copy,int lowerBound,int upperBound)
	{
		double firstValue;
		
		firstValue=copy.instance(lowerBound).classValue();
		for(int i=lowerBound+1;i<=upperBound;i++)
		{
			if (copy.instance(i).classValue()!=firstValue) return 2;
		}
		
		return 1;
	}
	
	/**
	 * @param lowerBound gives the index of the first element in the intervall
	 * @param upperBound gives the index of the last element in the intervall
	 * @param copy gives the already ordered instance set
	 * @return the Entropie on this set
	 */
	public double entropy(Instances copy,int lowerBound,int upperBound)
	{
					

		double countVal1=0;
		double countVal2=0;
		
		for(int i=lowerBound;i<=upperBound;i++)
		{
			if (copy.instance(i).classValue()==0) countVal1++;
			else countVal2++;
		}
		
		double probCV1=countVal1/(countVal1+countVal2);
		double probCV2=countVal2/(countVal1+countVal2);
		
		if (probCV1==0 || probCV2==0) return 1;
		
		return (-probCV1)*Utils.log2(probCV1)- probCV2*Utils.log2(probCV2);
	}
	
	public void setOutputFormat()
	{
		try 
		{
			FastVector attributes = new FastVector(getInputFormat().numAttributes());
			
			int listIndex=0;
			
			for(int i = 0; i < getInputFormat().numAttributes(); i++) {
				if (getInputFormat().attribute(i).isNumeric())
				{
					
					attributes.addElement(new Attribute(getInputFormat().
						      attribute(i).name(),determineNewAttributeType(listIndex)));
					listIndex++;
				}
				else
				{
					FastVector values=new FastVector();
					
					for(int e=0;e<getInputFormat().attribute(i).numValues();e++)
					{
						values.addElement(getInputFormat().attribute(i).value(e));
					}
					
					attributes.addElement(new Attribute(getInputFormat().
						      attribute(i).name(),values));
				}
			}
			
			Instances outputFormat = 
			      new Instances(getInputFormat().relationName(), attributes, 0);
			outputFormat.setClassIndex(getInputFormat().classIndex());
			setOutputFormat(outputFormat);	
		}
		catch (Exception ex) {
			System.out.println("Exception in setOutputFormat: "+ex.toString());
		    }
	}
	
	protected FastVector determineNewAttributeType(int listIndex)
	{
		FastVector values=new FastVector();
		
		TreeSet<Double> cutPoints=attributesCutPoints.get(listIndex);
		
		Iterator<Double> it=cutPoints.iterator();
		
		
		if (cutPoints.size()>0) {
			
			values.addElement("-inf < X <"+ Utils.doubleToString(cutPoints.first(), 3));
		
			double last=it.next();
			double actual;
			while(it.hasNext())
			{
				actual=it.next();
				values.addElement(Utils.doubleToString(last, 3) + " < X <"
						+ Utils.doubleToString(actual, 3) );
				last=actual;
			}
			values.addElement(Utils.doubleToString(last, 3)+ " < X < -inf");
			
		}
		else values.addElement("-inf < X < inf"); 
				
		return values;	
	}
	
	protected void convertInstance(Instance instance)
	{
		int listIndex=0;
		
		double [] values = new double [outputFormatPeek().numAttributes()];
		
		for(int i = 0; i < getInputFormat().numAttributes(); i++) {
			if(getInputFormat().attribute(i).isNumeric())
			{
				Iterator<Double> it=attributesCutPoints.get(listIndex).iterator();
				double actualValue=instance.value(i);
				int index=0;
				while(it.hasNext() && actualValue > it.next() ) index++;
				values[i]=index;
				
				listIndex++;
			}
			else values[i]=instance.value(i);
			
		}
		Instance newInst = new Instance(instance.weight(), values);
		
		newInst.setDataset(getOutputFormat());
		push(newInst);
	}
	
	public static void main(String [] argv) {

	    try {
	     ArffLoader loader=new ArffLoader();
	     File database=new File("spambase.ARFF");
	     loader.setSource(database);
	     
	     Instances inst=loader.getDataSet();
	     
	     Instances header=loader.getStructure();
	     header.setClassIndex(header.numAttributes()-1);
	  
	     
	     EntroDiscretize discr=new EntroDiscretize();
	     discr.setInputFormat(header);
	     
	     
	     
	     
	     for(int i=0;i<inst.numInstances();i++) 
	     {
	    	 discr.input(inst.instance(i));
	    	 //System.out.println(inst.instance(i).toString());
	     }
	     
	     discr.batchFinished();
	     
	     System.out.println(discr.output().toString());
	  
	      } catch (Exception ex) {
	      System.out.println(ex.getMessage());
	    }
	  }
}
