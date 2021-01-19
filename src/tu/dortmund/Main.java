package tu.dortmund;

import java.lang.System.Logger;
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
     * @param args No arguments necessary.
     */
    public static void main(String[] args) {
        int vocabularySize = 5000;
        int numTopics = 200;
        int[][] documents = generateRandomDocuments(500, 500, vocabularySize, 42);

        ArrayList<LdaModel> ldaSamplers = new ArrayList<LdaModel>();
        ldaSamplers.add(new GibbsLda(documents, vocabularySize, numTopics));
        ldaSamplers.add(new SparseLda(documents, vocabularySize, numTopics));
        ldaSamplers.add(new AliasLda(documents, vocabularySize, numTopics));
        ldaSamplers.add(new LightLda(documents, vocabularySize, numTopics));
        ldaSamplers.add(new FTreeLda(documents, vocabularySize, numTopics));

        for (LdaModel ldaSampler : ldaSamplers) {
            testLdaSampler(ldaSampler, 100, 20);
        }
    }

    /**
     * Generates random documents. A single document is represented as an integer
     * array. Each term in the vocabulary is represented by a unique integer. If for
     * example a document is [1,1,2,3] this indicates that the document contains 4
     * word tokes, where the first word appears twice in the document. This
     * representation is therefore NOT a Bag Of Words (BOW) format.
     * 
     * @param numDocuments       Number of documents that should be generated.
     * @param documentsMaxLength Maximum length of a document.
     * @param vocabularySize     Size of the Vocabulary (How many different terms
     *                           exist throughout all documents?).
     * @param seed               A seed value.
     * @return 2D integer array.
     */
    private static int[][] generateRandomDocuments(int numDocuments, int documentsMaxLength, int vocabularySize,
            int seed) {
        int[][] documents = new int[numDocuments][];
        Random random = new Random(seed);
        for (int i = 0; i < documents.length; i++) {
            documents[i] = random.ints(random.nextInt(documentsMaxLength) + 1, 0, vocabularySize).toArray();
        }

        return documents;
    }

    /**
     * Executes a LDAModel and measures the total time that the execution took.
     * 
     * @param ldaSampler The LDAModel that should be tested.
     * @param iterations Number of iterations.
     * @param llSteps    Number of iterations after which the method prints out the
     *                   current LogLikelihood of the model.
     */
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
