import java.util.ArrayList;
import java.util.Random;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.core.Measurement;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;

public class StreamKNN extends AbstractClassifier {

	private static final long serialVersionUID = 1L;

	// EJERCICIO 1. AÑADIR PROPIEDADES PUBLICAS Y PRIVADAS Y AÑADIR EL
	// CONSTRUCTOR

	// PARÁMETROS DEL ALGORITMO
	int k; // number of neighbors
	int maxSize; // max size of the window or reservoir
	boolean useReservoir = false; // whether it will be or not a reservoir

	// PROPIEDADES PRIVADAS
	private Instances window;  // The window of instances
	private int maxClassValue; // Used to count the number of different class values
	private int numProcessedInstances; // number of processed instances(for
										// reservoir sampling)

	// Seed for randomization/
	protected long seed = 1;	
	// Generator of random numbers */
	protected Random rand;

	// CONSTRUCTOR
	public StreamKNN(int k, int maxSize, boolean useReservoir) {
		this.k = k;
		this.maxSize = maxSize;
		this.useReservoir = useReservoir;
	}

	public void resetLearningImpl() {

		// Prepares the window
		int numAttributes = this.getModelContext().numAttributes();

		// Prepares an empty window
		ArrayList<Attribute> attributes = new ArrayList<Attribute>(numAttributes);
		for (int i = 0; i < numAttributes; i++) {
			attributes.add(this.getModelContext().attribute(i));
		}
		window = new Instances(this.getModelContext().relationName(), attributes, 0);
		window.setClassIndex(window.numAttributes() - 1);

		// EJERCICIO 2. INICIALIZAR maxClassValue y numInstances
		maxClassValue = -1; //¿¿Inicializar a -1 o a 0???
		numProcessedInstances = 0;
		
		rand = new Random(seed);
	}

	@Override
	public void trainOnInstanceImpl(Instance instance) {
		
		// Updates classCount
		if (instance.classValue() > maxClassValue)		
			maxClassValue = (int) instance.classValue();
		
		// The window is updated with the new instance
		if (!useReservoir)
			updateWindow(instance);
		else
			updateReservoir(instance);
		numProcessedInstances++;
	}

	private void updateWindow(Instance instance) {
		
		// EJERCICIO 3: ACTUALIZAR LA VENTANA
		if (window.numInstances() == maxSize)
			window.remove(0);
		window.add(instance);
	}

	private void updateReservoir(Instance instance) {
		
		// EJERCICIO 4. ACTUALIZAR EL RESERVOIR
		if (window.numInstances() == maxSize) {
			
			// Random number [0..numProcessedInstances-1]
			// the i-th instance has numProcessedInstances position			
			int aleatorio = rand.nextInt(numProcessedInstances);			
			 
			if (aleatorio < maxSize) {
				// Random number [0..window.size()-1]				
				window.remove(aleatorio);
				window.add(instance);
			}
		}
		else
		{	
		    window.add(instance);
		}
	}

	public double[] getVotesForInstance(Instance instance) {

		// EJERCICIO 5. IMPLEMENTAR LA BÚSQUEDA DE VECINOS
		// brute force search algorithm for nearest neighbor search
		LinearNNSearch search;
		search = new LinearNNSearch(window);
		//search.setSkipIdentical(true); //No funciona si utilizamos esto

		// KDTree search algorithm for nearest
		// neighbor search
		// search = new KDTree(); 
		// try {
		// search.setInstances(this.window);
		// } catch (Exception e1) {
		// // TODO Auto-generated catch block
		// e1.printStackTrace();
		// }

		Instances neighbours = null;

		// Returns k nearest instances in the current neighborhood to the
		// supplied instance
		try {
			//min(k, window.numInstances) in case there are not enough neighbours in the window
			neighbours = search.kNearestNeighbours(instance, Math.min(k, window.numInstances()));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// EJERCICIO 6. REALIZAR EL CONTEO DE LOS RESULTADOS DE LOS VECINOS
		// maxClassValue is the index of the class label therefore
		// the real number of classes will be maxClassValue+1
		double votes[] = new double[maxClassValue + 1];
		for (int i = 0; i < neighbours.numInstances(); i++)
			votes[(int) neighbours.instance(i).classValue()]++;
        
		for(int i =0; i<votes.length; i++)
			votes[i]/=neighbours.numInstances();
		
		return votes;
	}

	@Override
	public boolean isRandomizable() {	
		return true;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub

	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

}
