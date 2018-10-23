//--- by WEI@OHK 2018-10-20 ---//
package SparkJob;

import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Date;
import java.io.Serializable;
import java.util.Comparator;
import java.util.Collections;
import java.text.SimpleDateFormat;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;

import com.sun.jna.Library;
import com.sun.jna.Native;

import fr.inria.optimization.cmaes.CMAEvolutionStrategy;

/// A spark version of CMA_ES (maximize)
/// The CMA_ES library comes from https://www.lri.fr/~hansen/cmaes_inmatlab.html#java
public class CMA_ES {
	// Version numbers
	public final static int VERSION_MAJOR = 1;
	public final static int VERSION_MINOR = 2;

	// Randgen
	private static final Random randgen = new Random(System.currentTimeMillis());

	// Program entrance
	public static void main(String[] args) throws Exception {
		// Save job settings
		JobSettings settings = new JobSettings(args);

		// Date format	
		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

		//------------Log JobSettings--------------------------------------------------//
		FSDataOutputStream os_argsLog = FileSystem.get(new Configuration()).create(new Path(settings.getOutputPath() + "/jobSettings.log"), true);
		os_argsLog.write(("CMA_ESv" + VERSION_MAJOR + "." + VERSION_MINOR + " Application\n").getBytes("UTF-8"));
		for(String value : args) {
			os_argsLog.write((value+" ").getBytes("UTF-8"));
		}
		os_argsLog.write(("\n").getBytes("UTF-8"));
		os_argsLog.close();	
		//-----------------------------------------------------------------------------//

		// Evaluator initialization
		Evaluator evaluator = new Evaluator(settings.getNativeLib(), settings.getFuncIndex(), settings.getAddInfoLength());

		// Spark initialization
		JavaSparkContext sc = new JavaSparkContext(new SparkConf());

		// Initial values for CMA_ES
		int total_dimension = 0;
		for(int i : settings.getDimensions())
			total_dimension += i;	
		double[] initial_x = new double[total_dimension];
		double[] initial_sd = new double[total_dimension];
		double[] sd_ub = new double[total_dimension];
		double[] sd_lb = new double[total_dimension];

		// !!!Main Loop!!! //
		for(int i = 0; i < settings.getTrialsNum(); i++) {
			//------------Log fitness------------------------------------------------------//
			FSDataOutputStream os_fitness = FileSystem.get(new Configuration()).create(new Path(settings.getOutputPath() + "/Trial_" + i + "/fitness.log"), true);
			FSDataOutputStream os_overview = FileSystem.get(new Configuration()).create(new Path(settings.getOutputPath() + "/Trial_" + i + "/overview.log"), true);
			//-----------------------------------------------------------------------------//

			// CMA_ES initialization
			CMAEvolutionStrategy cma = new CMAEvolutionStrategy();
			cma.setDimension(total_dimension);
			int index = 0;
			for(int j = 0; j < settings.getDimensions().length; j++) {
				double ub = (settings.getUpperBounds()[j] - settings.getLowerBounds()[j]) * 0.3;
				double lb = (settings.getUpperBounds()[j] - settings.getLowerBounds()[j]) * 0.00001;
				for(int k = 0; k < settings.getDimensions()[j]; k++, index++) {
					initial_x[index] = (randgen.nextDouble() * 0.4 + 0.3) * (settings.getUpperBounds()[j] - settings.getLowerBounds()[j]) + settings.getLowerBounds()[j];
					initial_sd[index] = settings.getInitialsd()[j];
					sd_ub[index] = ub;
					sd_lb[index] = lb;
				}
			}
			cma.setInitialX(initial_x);
			cma.setInitialStandardDeviations(initial_sd);
			cma.options.upperStandardDeviations = sd_ub;
			cma.options.lowerStandardDeviations = sd_lb;

			// Set population size
			cma.parameters.setPopulationSize(settings.getLambda());
		
			// Stop conditions
			cma.options.stopFitness = Double.NEGATIVE_INFINITY;
			cma.options.stopTolFun = Double.MIN_VALUE;

			// Initialize cma and get fitness array to fill in later
			double[] fitness = cma.init();

			// Iteration loop for each trial
			for(int j = 0; j < settings.getMaxGeneration(); j++) {
				//------------Log overview-----------------------------------------------------//
				os_overview.write(("************ Generation_" + j + " started at " + df.format(new Date()) + " ************\n").getBytes("UTF-8"));
				os_overview.hflush();
				//-----------------------------------------------------------------------------//

				// Get a feasible population
				List<double[]> pop = Arrays.asList(cma.samplePopulation());
				try {
					int failure = 0;
					for(int k = 0; k < pop.size(); k++)
						while(!evaluator.isFeasible(pop.get(k), settings)) {
							pop.set(k, cma.resampleSingle(k));
							if(++failure > settings.getLambda()*100+10000) {
								//------------Log overview-----------------------------------------------------//
								os_overview.write(("## ERROR: Too many failures in feasibility check. ##\n").getBytes("UTF-8"));
								os_overview.hflush();
								//-----------------------------------------------------------------------------//
								throw new Exception(); 
							}	
						}
					//------------Log overview-----------------------------------------------------//
					if(failure > 0) {
						os_overview.write(("## WARNING: Feasibility check failures: " + failure + " ##\n").getBytes("UTF-8"));
						os_overview.hflush();
					}
					//-----------------------------------------------------------------------------//
				} catch(Exception e) {
					System.err.println("## ERROR: Too many failures in feasibility check. ##");
					System.exit(-1);
				}

				// Parallel evaluation
				JavaRDD<double[]> input = sc.parallelize(pop);
				JavaRDD<EvaluationResult> output = input.map(evaluator);
				List<EvaluationResult> c_fitness = output.collect();
				for(int k = 0; k < pop.size(); k++)
					fitness[k] = c_fitness.get(k).fitness * -1.0;

				//------------Log meanx--------------------------------------------------------//
				FSDataOutputStream os_mean_x = FileSystem.get(new Configuration()).create(new Path(settings.getOutputPath() + "/Trial_" + i + "/Generation_" + j + "/MeanX.individual"), true);
				double[] c_mean_x = cma.getMeanX();
				for(int k = 0; k < c_mean_x.length; k++) {
					os_mean_x.write(Double.toString(c_mean_x[k]).getBytes("UTF-8"));
					if(k < c_mean_x.length-1)
						os_mean_x.write((", ").getBytes("UTF-8"));
				}
				os_mean_x.write(("\n").getBytes("UTF-8"));
				os_mean_x.close();
				//-----------------------------------------------------------------------------//

				// Update search distribution
				cma.updateDistribution(fitness);

				//------------Log bestx--------------------------------------------------------//
				FSDataOutputStream os_best_x = FileSystem.get(new Configuration()).create(new Path(settings.getOutputPath() + "/Trial_" + i + "/Generation_" + j + "/BestX.individual"), true);
				double[] c_best_x = cma.getBestRecentX();
				for(int k = 0; k < c_best_x.length; k++) {
					os_best_x.write(Double.toString(c_best_x[k]).getBytes("UTF-8"));
					if(k < c_best_x.length-1)
						os_best_x.write((", ").getBytes("UTF-8"));
				}
				os_best_x.write(("\n").getBytes("UTF-8"));
				os_best_x.close();
                
				if(settings.getAddInfoLength() > 0) {
					List<EvaluationResult> list = new ArrayList<EvaluationResult>(c_fitness);
					Collections.sort(list, new Comparator<EvaluationResult>() {
						public int compare(EvaluationResult result_1, EvaluationResult result_2) {
							if(result_1.fitness > result_2.fitness) {
								return -1;
							} else if(result_1.fitness < result_2.fitness) {
								return 1;
							} else {
								return 0;
							}
						}
					});
                    
					FSDataOutputStream os_best_x_long = FileSystem.get(new Configuration()).create(new Path(settings.getOutputPath() + "/Trial_" + i + "/Generation_" + j + "/BestX.longValue"), true);
					for(int k = 0; k < settings.getAddInfoLength(); k++) {
						os_best_x_long.write(Long.toString(list.get(0).longValue[k]).getBytes("UTF-8"));
						os_best_x_long.write(("\n").getBytes("UTF-8"));
					}
					os_best_x_long.close();
                    
                    
					FSDataOutputStream os_best_x_double = FileSystem.get(new Configuration()).create(new Path(settings.getOutputPath() + "/Trial_" + i + "/Generation_" + j + "/BestX.doubleValue"), true);
					for(int k = 0; k < settings.getAddInfoLength(); k++) {
						os_best_x_double.write(Double.toString(list.get(0).doubleValue[k]).getBytes("UTF-8"));
						os_best_x_double.write(("\n").getBytes("UTF-8"));
					}
					os_best_x_double.close();
				}
				//-----------------------------------------------------------------------------//
				
				//------------Log current generation-------------------------------------------//
				//--log fitness--//
				double average_fitness = 0.0;
				for(double f : fitness) {
					average_fitness += f * -1.0;
				}
				average_fitness /= pop.size();
				os_fitness.write((j + "\t" + cma.getBestRecentFunctionValue() * -1.0 + "\t" + average_fitness + "\n").getBytes("UTF-8"));
				os_fitness.hflush();
				//--log overview--//
				if(cma.stopConditions.getNumber() != 0) {
					os_overview.write(("## WARNING: CMA_ES termination criterion met: ##\n").getBytes("UTF-8"));
					for(String s : cma.stopConditions.getMessages())
						os_overview.write(("## " + s + "\n").getBytes("UTF-8"));
					cma.stopConditions.clear();
				}
				os_overview.write(("Recent Best: " + cma.getBestRecentFunctionValue() * -1.0 + "\n").getBytes("UTF-8"));
				os_overview.write(("Recent Average: " + average_fitness + "\n").getBytes("UTF-8"));
				os_overview.write(("Best: " + cma.getBestFunctionValue() * -1.0 + " $$ in generation " + cma.getBestSolution().getEvaluationNumber() / settings.getLambda() + " $$\n\n").getBytes("UTF-8"));
				os_overview.hflush();
				//-----------------------------------------------------------------------------//
			}
			
			//------------Log close--------------------------------------------------------//
			os_fitness.close();
			os_overview.close();
			//-----------------------------------------------------------------------------//
		}
        
		//End of Program, problem occurred in Spark 2.x.x
		//System.exit(0);
	}


	/// Evaluator Class 
	private static class Evaluator implements Serializable, Function<double[], EvaluationResult> {
		// Constructor with job settings
		Evaluator(String lib_name, int func_index, int add_info_length) {
			this.libName = lib_name;
			this.funcIndex = func_index;

			this.addInfoLength = add_info_length;
		}

		// Used for map transformation
		public EvaluationResult call(double[] x) {
			NativeLibLoader.setNativeLib(libName);
			long[] long_buffer = new long[addInfoLength];
			double[] double_buffer = new double[addInfoLength];
			double fitness = NativeLibLoader.JOBNATIVELIB.INSTANCE.evaluateFcns(x, funcIndex, long_buffer, double_buffer, addInfoLength);
			EvaluationResult result = new EvaluationResult(fitness, long_buffer, double_buffer);
			return result;
		}

		// Check individual feasibility
		public boolean isFeasible(double[] individual, JobSettings settings) {
			int index = 0;
			for(int i = 0; i < settings.getDimensions().length; i++)
				for(int j = 0; j < settings.getDimensions()[i]; j++, index++)
					if(individual[index]>settings.getUpperBounds()[i] || individual[index]<settings.getLowerBounds()[i])
						return false;
			return true;
		}

		/// Native lib Loader
		private static class NativeLibLoader {
			// Set native library
			public static void setNativeLib(String lib_name) {
				NativeLibLoader.libName = lib_name;
			}

			// JNA interface
			public interface JOBNATIVELIB extends Library {
				JOBNATIVELIB INSTANCE = (JOBNATIVELIB)Native.loadLibrary(libName, JOBNATIVELIB.class);
				double evaluateFcns(double individual[], int func_index, long long_buffer[], double double_buffer[], int add_info_length);
			}

			private static String libName;
		}
	
		private final String libName;
		private final int funcIndex;
        
		private final int addInfoLength;
	}

	/// Evaluation Result Class
	private static class EvaluationResult implements Serializable {
		// Constructor with results
		EvaluationResult(double fitness, long[] long_buffer, double[] double_buffer) {
			this.fitness = fitness;
			this.longValue = long_buffer;
			this.doubleValue = double_buffer;
		}
        
		// Results
		public double fitness;
		public long[] longValue;
		public double[] doubleValue;
	}

	/// Job Settings Class
	private static class JobSettings {
		// Default constructor
		JobSettings() {
			outputPath = "CMA_ESv" + VERSION_MAJOR + "." + VERSION_MINOR + "_output_" + System.currentTimeMillis();
			nativeLib = "";
			funcIndex = 0;

			trialsNum = 10;
			maxGeneration = 500;
			lambda = 200;

			dimensions = new int[1];
			upperBounds = new double[1];			
			lowerBounds = new double[1];			
			initialsd = new double[1];			
			
			dimensions[0] = 300;
			upperBounds[0] = 1.0;
			lowerBounds[0] = -1.0;
			initialsd[0] = 0.5;
			addInfoLength = 0;
		}

		// Constructor with args
		JobSettings(String[] args) throws Exception {
			// Call default constructor
			this();
			
			// Specified args in command line
			try {		
				for(int i = 0; i < args.length; i++) {
					switch(args[i]) {
						case "-OUTPUTPATH":
							setOutputPath(args[++i]);
							break;
						case "-LIB":
							setNativeLib(args[++i]);
							break;
						case "-FUNC_INDEX":
							setFuncIndex(Integer.parseInt(args[++i]));
							break;
						case "-TRIALS":
							setTrialsNum(Integer.parseInt(args[++i]));
							break;
						case "-G":
							setMaxGeneration(Integer.parseInt(args[++i]));
							break;
						case "-LAMBDA":
							setLambda(Integer.parseInt(args[++i]));
							break;
						case "-D_FORMAT":
							int D_COUNT = Integer.parseInt(args[++i]);
							dimensions = new int[D_COUNT];
							upperBounds = new double[D_COUNT];
							lowerBounds = new double[D_COUNT];
							initialsd = new double[D_COUNT];
							for(int j = 0; j < D_COUNT; j++) {
								String[] D_CONF = args[++i].split(",");
								if(D_CONF.length != 4) {
									System.err.println("Bad format for option 'D_FORMAT' ");
									throw new Exception();
								}
								dimensions[j] = Integer.parseInt(D_CONF[0]);
								upperBounds[j] = Double.parseDouble(D_CONF[1]);
								lowerBounds[j] = Double.parseDouble(D_CONF[2]);
								initialsd[j] = Double.parseDouble(D_CONF[3]);
							}
							break;
						case "-ADD_INFO_LENGTH":
							setAddInfoLength(Integer.parseInt(args[++i]));
							break;
						default:
							System.err.println("Undefined option: '" + args[i] + "'");
							throw new Exception(); 
					}
				}		
			} catch(Exception e) {
				System.err.println("Unknown command");
				System.err.println("Usage: spark-submit --num-executors NUM --files LIBNAME --name APPNAME CMA_ESv" + VERSION_MAJOR + "." + VERSION_MINOR + ".jar [<option>..]");
				System.err.println("Options (* must be specified):");
				System.err.println("-OUTPUT_PATH <PATH>         : Path for outputs.");
				System.err.println("-LIB <NAME> (*)             : Name of native library.");
				System.err.println("-FUNC_INDEX <NUM>           : Index of function in native library to be used.");
				System.err.println("-TRIALS <NUM>               : Number of trials");
				System.err.println("-G <NUM>                    : Max generation.");
				System.err.println("-LAMBDA <NUM>               : Population size.");
				System.err.println("-D_FORMAT <@SEE SAMPLES>    : Describe the individual.");
				System.err.println("-ADD_INFO_LENGTH <NUM>      : Length of additional information for each individual.");
				
				System.exit(-1);
			}
		}
		
		// Spark Task Parameters
		private String outputPath;		//path for outputs
		private String nativeLib;		//native library used for evaluation
		private int funcIndex;			//index of function in native library to be used
        
		// CMA_ES Parameters
		private int trialsNum;			//number of trials
		private int maxGeneration;		//max generation
		private int lambda;			//population size

		// Individual Parameters
		private int[] dimensions;		//dimensions
		private double[] upperBounds;		//upper_bounds of individual
		private double[] lowerBounds;		//lower_bounds of individual
		private double[] initialsd;		//initial standard deviation
		private int addInfoLength;		//length of additional information per individual
		
		// Getters
		public String getOutputPath() {
			return outputPath;
		}
		public String getNativeLib() {
			return nativeLib;
		}
		public int getFuncIndex() {
			return funcIndex;
		}
		public int getTrialsNum() {
			return trialsNum;
		}
		public int getMaxGeneration() {
			return maxGeneration;
		}
		public int getLambda() {
			return lambda;
		}
		public int[] getDimensions() {
			return dimensions;
		}
		public double[] getUpperBounds() {
			return upperBounds;
		}
		public double[] getLowerBounds() {
			return lowerBounds;
		}
		public double[] getInitialsd() {
			return initialsd;
		}
		public int getAddInfoLength() {
			return addInfoLength;
		}
		
		// Setters
		public void setOutputPath(final String value) {
			outputPath = value;
		}
		public void setNativeLib(final String value) {
			nativeLib = value;
		}
		public void setFuncIndex(final int value) {
			funcIndex = value;
		}
		public void setTrialsNum(final int value) {
			trialsNum = value;
		}
		public void setMaxGeneration(final int value) {
			maxGeneration = value;
		}
		public void setLambda(final int value) {
			lambda = value;
		}
		public void setDimensions(final int[] value) {
			dimensions = value;
		}
		public void setUpperBounds(final double[] value) {
			upperBounds = value;
		}
		public void setLowerBounds(final double[] value) {
			lowerBounds = value;
		}
		public void setInitialsd(final double[] value) {
			initialsd = value;
		}
		public void setAddInfoLength(final int value) {
			addInfoLength = value;
		}
	}
}
