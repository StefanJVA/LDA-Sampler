package tu.dortmund.lda.sampler;

import java.util.Arrays;
import java.util.SplittableRandom;

import tu.dortmund.lda.LdaModel;

/**
 * The normal Gibbs Sampling algorithm
 */
public class GibbsLda implements LdaModel {

    protected final int vocabularySize;
    protected final int numTopics;

    protected double[] alpha;
    protected double alphaSum;
    protected double[] beta;
    protected double betaSum;

    /**
     * All existing documents. A single document is represented as an integer array.
     * Each term in the vocabulary is represented by a unique integer. If for
     * example a document is [1,1,2,3] this indicates that the document contains 4
     * word tokes, where the first word appears twice in the document. This
     * representation is therefore NOT a Bag Of Words (BOW) format.
     */
    protected int[][] documents;

    /**
     * Holds the topic assignment for each word token in each document. It therefore
     * has the same dimensions as the documents 2d array.
     */
    protected int[][] matZ;

    /**
     * Count matrix that holds the number of times a word is assinged to a topic
     * throughout the whole corpus. For example matTopicWord[i][j] tells how often
     * word j is assigned to topic i. The matrix dimensions are therefore: number of
     * topics X vocabulary size.
     */
    protected int[][] matTopicWord;

    /**
     * Count matrix that holds the number of times a topic is used inside a
     * document. For example matDocTopic[i][j] tells how often a word token inside
     * document i is assigned to topic j. The matrix dimensions are therefore:
     * number of documents X number of topics.
     */
    protected int[][] matDocTopic;

    /**
     * Count vector that holds the total assignments of each topic throughout the
     * whole corpus.
     */
    protected int[] vecTopic;

    /**
     * Total number of word tokens throughout the whole corpus.
     */
    protected int numTokens;

    protected SplittableRandom random;

    /**
     * Constructor. Sets some default values for alpha and beta prior.
     * 
     * @param documents      The documents decoded as integers.
     * @param vocabularySize Size of the vocabulary.
     * @param numTopics      Number of topics that should be learned.
     */
    public GibbsLda(int[][] documents, int vocabularySize, int numTopics) {
        this.documents = documents;
        this.vocabularySize = vocabularySize;
        this.numTopics = numTopics;

        // set some default values for alpha (50 / numTopics)
        this.alpha = new double[numTopics];
        Arrays.fill(this.alpha, 50.0 / (double) numTopics);
        this.alphaSum = Arrays.stream(this.alpha).sum();

        // set some default values for beta (0.01)
        this.beta = new double[vocabularySize];
        Arrays.fill(this.beta, 0.01);
        this.betaSum = Arrays.stream(this.beta).sum();

        this.random = new SplittableRandom();
    }

    @Override
    public void initialize() {
        this.initializeCountMatrices();
    }

    /**
     * Initializes the count matrices. This may allocate a lot of main memory.
     */
    protected void initializeCountMatrices() {
        matTopicWord = new int[numTopics][vocabularySize];
        matDocTopic = new int[documents.length][numTopics];
        vecTopic = new int[numTopics];
        matZ = new int[documents.length][];
        numTokens = 0;

        for (int document = 0; document < documents.length; document++) {
            matZ[document] = new int[documents[document].length];
            for (int token = 0; token < documents[document].length; token++) {
                int word = documents[document][token];
                int topic = random.nextInt(numTopics); // random topic assignment
                matZ[document][token] = topic;
                matTopicWord[topic][word]++;
                matDocTopic[document][topic]++;
                vecTopic[topic]++;
                numTokens++;
            }
        }
    }

    @Override
    public void run(int iterations) {
        for (int i = 0; i < iterations; i++) {
            fullCorpusSweep();
        }
    }

    /**
     * The Gibbs sampling algorithm which iterates over every word token in every
     * document ones.
     */
    protected void fullCorpusSweep() {
        double[] cumulativeProbabilities = new double[numTopics];

        for (int document = 0; document < matZ.length; document++) {
            for (int token = 0; token < matZ[document].length; token++) {
                int topic = matZ[document][token];
                int word = documents[document][token];

                decrementCountMatrices(document, word, topic);

                double probabilitySum = 0.0;
                for (int topicI = 0; topicI < numTopics; topicI++) {
                    double probabilityOfTopicI = matDocTopic[document][topicI] + alpha[topicI];
                    double probabilityOfWordInTopicI = (matTopicWord[topicI][word] + beta[word])
                            / (vecTopic[topicI] + betaSum);
                    probabilitySum += probabilityOfTopicI * probabilityOfWordInTopicI;
                    cumulativeProbabilities[topicI] = probabilitySum;
                }
                double u = random.nextDouble() * probabilitySum;
                topic = lowerBound(cumulativeProbabilities, cumulativeProbabilities.length, u);

                incrementCountMatrices(document, word, topic);

                matZ[document][token] = topic;
            }
        }
    }

    /**
     * Decrements all count matrices by one for the given indices
     * 
     * @param document Document index
     * @param word     Word index
     * @param topic    Topic index
     */
    protected void decrementCountMatrices(int document, int word, int topic) {
        matTopicWord[topic][word]--;
        matDocTopic[document][topic]--;
        vecTopic[topic]--;
    }

    /**
     * Increments all count matrices by one for the given indices
     * 
     * @param document Document index
     * @param word     Word index
     * @param topic    Topic index
     */
    protected void incrementCountMatrices(int document, int word, int topic) {
        matTopicWord[topic][word]++;
        matDocTopic[document][topic]++;
        vecTopic[topic]++;
    }

    /**
     * Computes the current LogLikelihood of the model. This method takes some
     * computing resources!
     * 
     * @return LogLikelihood of the model.
     */
    @Override
    public double getLogLikelihood() {
        return getLogLikelihoodMallet();
    }

    /**
     * Compute the LogLikelihood of the model as suggested by the Mallet LDA
     * algorithm.
     * 
     * @return LogLikelihood of the model.
     */
    public double getLogLikelihoodMallet() {
        double logLikelihood = 0.0;

        // document topic
        double[] logGammaAlpha = new double[numTopics];
        for (int topic = 0; topic < numTopics; topic++) {
            logGammaAlpha[topic] = logGammaStirling(alpha[topic]);
        }
        for (int document = 0; document < documents.length; document++) {
            for (int topic = 0; topic < numTopics; topic++) {
                float count = matDocTopic[document][topic];
                if (count > 0.1) {
                    logLikelihood += logGammaStirling(alpha[topic] + count) - logGammaAlpha[topic];
                }
            }
            logLikelihood -= logGammaStirling(alphaSum + documents[document].length);
        }
        logLikelihood += documents.length * logGammaStirling(alphaSum);

        // topic term
        double[] logGammaBeta = new double[vocabularySize];
        for (int term = 0; term < vocabularySize; term++) {
            logGammaBeta[term] = logGammaStirling(beta[term]);
        }
        for (int topic = 0; topic < numTopics; topic++) {
            for (int term = 0; term < vocabularySize; term++) {
                float count = matTopicWord[topic][term];
                if (count > 0.1) {
                    logLikelihood += logGammaStirling(beta[term] + count) - logGammaBeta[term];
                }
            }
            logLikelihood -= logGammaStirling(betaSum + vecTopic[topic]);
        }
        logLikelihood += numTopics * logGammaStirling(betaSum);

        return logLikelihood;
    }

    public static final double HALF_LOG_TWO_PI = Math.log(2 * Math.PI) / 2;

    /**
     * Use a fifth order Stirling's approximation. Copied from Mallet LDA algorithm.
     * 
     * @param z Note that Stirling's approximation is increasingly unstable as z
     *          approaches 0. If z is less than 2, we shift it up, calculate the
     *          approximation, and then shift the answer back down.
     */
    public static double logGammaStirling(double z) {
        int shift = 0;
        while (z < 2) {
            z++;
            shift++;
        }

        double result = HALF_LOG_TWO_PI + (z - 0.5) * Math.log(z) - z + 1 / (12 * z) - 1 / (360 * z * z * z)
                + 1 / (1260 * z * z * z * z * z);

        while (shift > 0) {
            shift--;
            z--;
            result -= Math.log(z);
        }

        return result;
    }

    /**
     * Simple lower bound algorithm, which finds the lower bound in log(n)
     * complexity.
     * 
     * @param p      Sorted array (increasing order) of values for which the lower
     *               bound should be found.
     * @param length Values in p that appear after the length are ignored. The lower
     *               bound therefore needs to be between 0 and length.
     * @param u      The lower bound.
     * @return Index of the lower bound.
     */
    protected int lowerBound(double[] p, int length, double u) {
        int low = 0;
        int high = length;
        while (low < high) {
            int mid = (low + high) / 2;
            if (u <= p[mid]) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }

    /**
     * Returns the document topic distributions theta. Every row is a normalized
     * probability distribution that contains the probabilities for each topic
     * inside the corresponding document. For example if half of all word tokens in
     * document i are assigned to topic j, theta[i][j] contains the value 0.5 . This
     * method takes some computing resources!
     */
    @Override
    public double[][] getTheta() {
        double[][] theta = new double[documents.length][numTopics];

        for (int document = 0; document < documents.length; document++) {
            for (int topic = 0; topic < numTopics; topic++) {
                theta[document][topic] = (matDocTopic[document][topic] + alpha[topic])
                        / (documents[document].length + alphaSum);
            }
        }

        return theta;
    }

    /**
     * Returns the topic term distribution phi. Every row is a normalized
     * probability distribution that contains the word probabilities of the
     * corresponding topic. For example if half of all word tokens that are assigned
     * to topic i contain the term j, phi[i][j] contains the value 0.5 .
     */
    @Override
    public double[][] getPhi() {
        double[][] phi = new double[numTopics][vocabularySize];

        for (int topic = 0; topic < numTopics; topic++) {
            for (int word = 0; word < vocabularySize; word++) {
                phi[topic][word] = (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);
            }
        }

        return phi;
    }

    @Override
    public int[][] getMatZ() {
        return this.matZ;
    }

    @Override
    public int getNumTokens() {
        return numTokens;
    }

    @Override
    public double[] getAlpha() {
        return alpha;
    }

    @Override
    public double[] getBeta() {
        return beta;
    }

    @Override
    public int getNumTopics() {
        return numTopics;
    }

    @Override
    public int getVocabularySize() {
        return vocabularySize;
    }

    @Override
    public void setSeed(long seed) {
        this.random = new SplittableRandom(seed);
    }

    @Override
    public void setAlpha(double[] alpha) {
        this.alpha = alpha;
        this.alphaSum = Arrays.stream(alpha).sum();
    }

    @Override
    public void setBeta(double[] beta) {
        this.beta = beta;
        this.betaSum = Arrays.stream(beta).sum();
    }

    @Override
    public void setAlpha(double alphaSymmetric) {
        this.alphaSum = 0.0;
        for (int i = 0; i < this.alpha.length; i++) {
            this.alpha[i] = alphaSymmetric;
            this.alphaSum += alphaSymmetric;
        }
    }

    @Override
    public void setBeta(double betaSymmetric) {
        this.betaSum = 0.0;
        for (int i = 0; i < this.beta.length; i++) {
            this.beta[i] = betaSymmetric;
            this.betaSum += betaSymmetric;
        }
    }
}
