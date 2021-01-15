package tu.dortmund;

import java.util.ArrayList;
import java.util.Random;

import tu.dortmund.lda.LdaModel;
import tu.dortmund.lda.sampler.*;


/**
 * Main class.
 */
public class Main {
    
    /**
     * Application entry point.
     * 
     * @param args Application arguments.
     */
    public static void main(String[] args) {
        int vocabularySize = 1000;
        int[][] documents = generateRandomDocuments(100, 1000, vocabularySize, 42);
        int numNormalTopics = 100;

        ArrayList<LdaModel> ldaSamplers = new ArrayList<LdaModel>();
        ldaSamplers.add(new GibbsLda(documents, vocabularySize, numNormalTopics));
        ldaSamplers.add(new SparseLda(documents, vocabularySize, numNormalTopics));
        ldaSamplers.add(new AliasLda(documents, vocabularySize, numNormalTopics));
        ldaSamplers.add(new FTreeLda(documents, vocabularySize, numNormalTopics));

        for (LdaModel ldaSampler : ldaSamplers) {
            testLdaSampler(ldaSampler, 200, 100);
        }        
    }

    private static int[][] generateRandomDocuments(int numDocuments, int documentsMaxLength, int vocabularySize, int seed){
        int[][] documents = new int[numDocuments][];
        Random random = new Random(seed);
        for (int i = 0; i < documents.length; i++) {
            documents[i] = random.ints(random.nextInt(documentsMaxLength)+1, 0, vocabularySize).toArray();
        }

        return documents;
    }

    private static void testLdaSampler(LdaModel ldaSampler, int iterations, int llSteps) {
        System.out.println("Testing LDA Sampler with name: " + ldaSampler.getClass().getSimpleName());

        ldaSampler.initialize();

        long totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            if (i % llSteps == 0) {
                System.out.println("LogLikelihood after " + i + " iterations: " + ldaSampler.getLogLikelihood());
            }
            long startTime = System.currentTimeMillis();
            ldaSampler.run(1);
            long endTime = System.currentTimeMillis();

            long deltaTime = endTime - startTime;
            totalTime += deltaTime;

            
        }
        System.out.println("LogLikelihood after " + iterations + " iterations: " + ldaSampler.getLogLikelihood());
        System.out.println("Execution took " + totalTime / 1000 + " seconds");
    }
}

