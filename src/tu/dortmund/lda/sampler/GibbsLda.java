package tu.dortmund.lda.sampler;

import tu.dortmund.lda.LdaModel;

import java.text.DecimalFormat;
import java.util.*;

public class GibbsLda implements LdaModel {

    protected final int vocabularySize;
    protected int[][] documents;
    protected double[] alpha;
    protected double alphaSum;
    protected double[] beta;
    protected double betaSum;
    protected final int numNormalTopics;
    protected int numTotalTopics;

    protected int[][] matZ;
    protected float[][] matTopicWord;
    protected float[][] matDocTopic;
    protected float[] vecTopic;

    protected int numTokens;

    protected SplittableRandom random;

    public GibbsLda(int[][] documents, int vocabularySize, int numNormalTopics){
        this.documents = documents;
        this.vocabularySize = vocabularySize;
        this.numNormalTopics = numNormalTopics;
        this.numTotalTopics = numNormalTopics;

        //set some default values for alpha and beta
        this.alpha = new double[numNormalTopics];
        for (int i = 0; i < alpha.length; i++) {
            this.alpha[i] = 50.0 / (double) numNormalTopics;
        }
        this.alphaSum = Arrays.stream(alpha).sum();

        this.beta = new double[vocabularySize];
        for (int i = 0; i < beta.length; i++) {
            this.beta[i] = 0.01;
        }
        this.betaSum = Arrays.stream(beta).sum();


        this.random = new SplittableRandom(42l);
    }

    @Override
    public void initialize() {
        this.initializeCountMatrices();
    }

    protected void initializeCountMatrices(){
        matTopicWord = new float[numTotalTopics][vocabularySize];
        matDocTopic = new float[documents.length][numTotalTopics];
        vecTopic = new float[numTotalTopics];
        matZ = new int[documents.length][];
        numTokens = 0;

        for(int document = 0; document < documents.length; document++){
            matZ[document] = new int[documents[document].length];
            for (int token = 0; token < documents[document].length; token++){
                int word = documents[document][token];
                int topic = sampleTopicForWord();
                matZ[document][token] = topic;
                matTopicWord[topic][word]++;
                matDocTopic[document][topic]++;
                vecTopic[topic]++;
                numTokens++;
            }
        }
    }

    protected int[] computeWordCount() {
        int[] vecWordCount = new int[vocabularySize];
        for(int document = 0; document < documents.length; document++){
            for (int token = 0; token < documents[document].length; token++){
                vecWordCount[documents[document][token]]++;
            }
        }
        return vecWordCount;
    }

    protected int sampleTopicForWord() {
        return random.nextInt(numTotalTopics);
    }

    @Override
    public void run(int iterations){
        for (int i = 0; i < iterations; i++) {
            fullCorpusSweep();
            //showProgressInConsole((int)((double)i / (double)iterations * 100.0));
        }
    }

    protected void fullCorpusSweep() {
        double[] p = new double[numTotalTopics];

        for (int document = 0; document < matZ.length; document++) {
            for (int token = 0; token < matZ[document].length; token++) {
                int topic = matZ[document][token];
                int word = documents[document][token];

                decrementCountMatrices(document, word, topic);

                double psum = 0.0;
                for (int topicI = 0; topicI < numNormalTopics; topicI++) {
                    double propOfTopicI = matDocTopic[document][topicI] + alpha[topicI];
                    double propOfWordInTopicI = (matTopicWord[topicI][word] + beta[word]) / (vecTopic[topicI] + betaSum);
                    psum += propOfTopicI * propOfWordInTopicI;
                    p[topicI] = psum;
                }
                double u = random.nextDouble() * psum;
                topic = lowerBound(p, p.length, u);

                incrementCountMatrices(document, word, topic);

                matZ[document][token] = topic;
            }
        }
    }

    protected int sampleUnnormalizedDistribution(double[] p) {
        //cumulative method
        for (int i = 1; i < p.length; i++) {
            p[i] += p[i - 1];
        }

        double u = random.nextDouble() * p[p.length - 1];
        int result = 0;
        for (result = 0; result < p.length; result++) {
            if (u < p[result])
                break;
        }

        return result;
    }

    protected void decrementCountMatrices(int document, int word, int topic){
        matTopicWord[topic][word]--;
        matDocTopic[document][topic]--;
        vecTopic[topic]--;
    }

    protected void incrementCountMatrices(int document, int word, int topic){
        matTopicWord[topic][word]++;
        matDocTopic[document][topic]++;
        vecTopic[topic]++;
    }

    @Override
    public double getLogLikelihood() {
        return getLogLikelihoodMallet();
    }

    public double getLogLikelihoodFull() {
        double[][] theta = this.getTheta();
        double[][] phi = this.getPhi();

        double logLikelihood = 0.0;
        for (int d = 0; d < documents.length; d++) {
            for (int wi = 0; wi < documents[d].length; wi++) {
                double wordProp = 0.0;
                for (int t = 0; t < numTotalTopics; t++) {
                   wordProp += theta[d][t] * phi[t][documents[d][wi]];
                }
                logLikelihood += Math.log(wordProp);
            }
        }

        return logLikelihood;
    }

    public double getLogLikelihoodDMLC() {
        double[][] theta = this.getTheta();
        double[][] phi = this.getPhi();

        int num_tokens = 0;
        double sum = 0.0;
        for (int d = 0; d < documents.length; d++) {
            double dsum = 0.0;
            num_tokens += documents.length;
            for (int wi = 0; wi < documents[d].length; wi++) {
                int word = documents[d][wi];
                double wsum = 0.0;
                for (int t = 0; t < numTotalTopics; t++) {
                    wsum += (theta[d][t] ) * phi[t][word];
                }
                dsum += Math.log(wsum);
            }
            sum += dsum - (documents.length * Math.log(documents.length + numTotalTopics));
        }

        return sum / num_tokens;
    }

    public double getLogLikelihoodMallet() {
        double logLikelihood = 0.0;

        // document topic
        double[] logGammaAlpha = new double[numTotalTopics];
        for (int topic = 0; topic < numTotalTopics; topic++) {
            logGammaAlpha[topic] = logGammaStirling(alpha[topic]);
        }
        for (int document = 0; document < documents.length; document++) {
            for (int topic = 0; topic < numTotalTopics; topic++) {
                float count = matDocTopic[document][topic];
                if (count > 0.1){
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
        for (int topic = 0; topic < numTotalTopics; topic++) {
            for (int term = 0; term < vocabularySize; term++) {
                float count = matTopicWord[topic][term];
                if(count > 0.1){
                    logLikelihood += logGammaStirling(beta[term] + count) - logGammaBeta[term];
                }
            }
            logLikelihood -= logGammaStirling(betaSum + vecTopic[topic]);
        }
        logLikelihood += numTotalTopics * logGammaStirling(betaSum);

        return logLikelihood;
    }

    public static final double HALF_LOG_TWO_PI = Math.log(2 * Math.PI) / 2;

    /** Use a fifth order Stirling's approximation.
     * Copied from mallet
     *
     *	@param z Note that Stirling's approximation is increasingly unstable as z approaches 0.
     *          	If z is less than 2, we shift it up, calculate the approximation, and then shift the answer back down.
     */
    public static double logGammaStirling(double z) {
        int shift = 0;
        while (z < 2) {
            z++;
            shift++;
        }

        double result = HALF_LOG_TWO_PI + (z - 0.5) * Math.log(z) - z +
                1/(12 * z) - 1 / (360 * z * z * z) + 1 / (1260 * z * z * z * z * z);

        while (shift > 0) {
            shift--;
            z--;
            result -= Math.log(z);
        }

        return result;
    }

    @Override
    public double[][] getTheta() {
        double[][] theta = new double[documents.length][numTotalTopics];

        for (int document = 0; document < documents.length; document++) {
            for (int topic = 0; topic < numTotalTopics; topic++) {
//                theta[document][topic] = (matDocTopic[document][topic] + alpha[topic]) / (documents[document].length + numTotalTopics * alpha);
                theta[document][topic] = (matDocTopic[document][topic] + alpha[topic]) / (documents[document].length + alphaSum);
            }
        }

        return theta;
    }

    @Override
    public double[][] getPhi() {
        double[][] phi = new double[numTotalTopics][vocabularySize];

        for (int topic = 0; topic < numTotalTopics; topic++) {
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

    protected int[] computeTermDocumentFrequency(){
        int[] tdf = new int[vocabularySize];
        for (int document = 0; document < documents.length; document++) {
            int[] distinctWords = java.util.stream.IntStream.of(documents[document]).distinct().toArray();
            for (int word : distinctWords) {
                tdf[word]++;
            }
        }
        return tdf;
    }

    protected int[][] computeBagOfWords(){
        int[][] bow = new int[documents.length][vocabularySize];
        for (int d = 0; d < documents.length; d++) {
            for (int wi = 0; wi < documents[d].length; wi++) {
                int word = documents[d][wi];
                bow[d][word]++;
            }
        }
        return bow;
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
    public void setBeta(double[] beta){
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
    public void setBeta(double betaSymmetric){
        this.betaSum = 0.0;
        for (int i = 0; i < this.beta.length; i++) {
            this.beta[i] = betaSymmetric;
            this.betaSum += betaSymmetric;
        }
    }

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

    public void removeWords(ArrayList<Integer> removalWords) {
        for (int document = 0; document < documents.length; document++) {
//            List<Integer> newDocument = Arrays.stream(documents[document]).boxed().collect(Collectors.toList());
//            List<Integer> newTopicAssignments = Arrays.stream(z[document]).boxed().collect(Collectors.toList());
//            //newDocument.removeIf()
//
//            ListIterator<Integer> iter = newDocument.listIterator();
//            while(iter.hasNext()){
//                int word = iter.next();
//                if(removalWords.contains(word)){
//                    iter.remove();
//                }
//            }
            ArrayList<Integer> removedIndices = new ArrayList<>();
            for (int wi = 0; wi < documents[document].length; wi++) {
                int topic = matZ[document][wi];
                int word = documents[document][wi];
                if(removalWords.contains(word)){
                    decrementCountMatrices(document, word, topic);
                    removedIndices.add(wi);
                }
            }

            int index = 0;
            int[] newDocument = new int[documents[document].length - removedIndices.size()];
            int[] newTopicAssignments = new int[documents[document].length - removedIndices.size()];
            for (int wi = 0; wi < documents[document].length; wi++) {
                if(!removedIndices.contains(wi)){
                    newDocument[index] = documents[document][wi];
                    newTopicAssignments[index] = matZ[document][wi];
                    index++;
                }
            }

            documents[document] = newDocument;
            matZ[document] = newTopicAssignments;
        }
    }

    @Override
    public int[] getAssignmentsPerTopic() {
        int[] totalAssis = new int[vecTopic.length];
        for (int i = 0; i < vecTopic.length; i++) {
            totalAssis[i] = (int)(vecTopic[i]+0.1);
        }
        return totalAssis;
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
    public int getNumNormalTopics() {
        return numNormalTopics;
    }

    @Override
    public int getNumTotalTopics() {
        return numTotalTopics;
    }

    @Override
    public int getVocabularySize() {
        return vocabularySize;
    }
}
