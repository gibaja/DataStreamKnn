
import weka.core.Instance;
import moa.streams.generators.RandomRBFGenerator;


public class Ejecutar {

	public static void main(String[] args) {
		
		//Prepares the data generator
		int numInstances = 10000;	
		RandomRBFGenerator stream = new RandomRBFGenerator();
		stream.prepareForUse();
		
		//Prepares the classifier
		StreamKNN knn = new StreamKNN(10, 1000, true); 
		
		knn.setModelContext(stream.getHeader());//Sets the reference to the header of the data stream
        knn.prepareForUse();
        
        long time_start, time_end;  
        
        double accuracy;
        int correct=0, samples =0;
        time_start = System.currentTimeMillis();
        while(stream.hasMoreInstances() & samples<numInstances)
        {
        	Instance inst = stream.nextInstance();
        	if(knn.correctlyClassifies(inst)) //Utiliza getVotesForInstance 
        	  correct++;	
        	samples++;
        	knn.trainOnInstance(inst);
        	accuracy  = 100.0*(double)correct/(double)samples;
        	System.out.println("#samples: "+samples+ " #correct: "+correct+" accuracy: "+accuracy);
        }	
        time_end = System.currentTimeMillis();	    
        System.out.println("the task has taken "+ ( time_end - time_start ) +" milliseconds");
	}

}
