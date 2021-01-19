package tu.dortmund.lda.sampler;

import tu.dortmund.lda.data_structure.AliasTable;

import java.util.ArrayList;

public class AliasLda extends GibbsLda {

    static final int MH_STEPS = 2;

    AliasTable[] aliasTables;
    private ArrayList<Integer>[] nonzeroDocTopic;

    public AliasLda(int[][] documents, int vocabularySize, int k) {
        super(documents, vocabularySize, k);
    }

    @Override
    public void initialize() {
        super.initializeCountMatrices();

        this.aliasTables = new AliasTable[vocabularySize];
        for (int w = 0; w < vocabularySize; w++) {
            aliasTables[w] = new AliasTable(numTopics);
            updateAliasTable(w);
        }

        // build up nonzero documents data structure
        nonzeroDocTopic = new ArrayList[documents.length];
        for (int document = 0; document < documents.length; document++) {
            ArrayList<Integer> nonzeroList = new ArrayList<Integer>();
            for (int topic = 0; topic < numTopics; topic++) {
                if(matDocTopic[document][topic] > 0 && !nonzeroList.contains(topic)) {
                    nonzeroList.add(topic);
                }
            }
            nonzeroDocTopic[document] = nonzeroList;
        }
    }

    private void updateAliasTable(int word) {
        AliasTable at = aliasTables[word];
        at.resetSampleCount();
        double psum = 0.0;
        for (int t = 0; t < numTopics; t++) {
            double tmp = alpha[t] * (matTopicWord[t][word] + beta[word]) / (vecTopic[t] + betaSum);
            at.getUnnormalizedProbability()[t] = tmp;
            psum += tmp;
        }
        at.setProbabilitySum(psum);
        at.construct();
    }

    @Override
    protected void fullCorpusSweep() {
        double[] pdw = new double[numTopics];

        for (int document = 0; document < documents.length; document++) {
            for (int wi = 0; wi < documents[document].length; wi++) {
                int word = documents[document][wi];
                int topic = matZ[document][wi];

                decrementCountMatrices(document, word, topic);
                if(matDocTopic[document][topic] == 0){
                    nonzeroDocTopic[document].remove(Integer.valueOf(topic));
                }

                ArrayList<Integer> nonzeroTopics = nonzeroDocTopic[document];
                double pdwSum = 0.0;
                for (int i = 0; i < nonzeroDocTopic[document].size(); i++) {
                    int t = nonzeroTopics.get(i);
                    pdwSum += matDocTopic[document][t] * (matTopicWord[t][word] + beta[word]) / (vecTopic[t] + betaSum);
                    pdw[i] = pdwSum;
                }

                AliasTable wordTable = aliasTables[word];

                int newTopic = -1;
                for (int mhStep = 0; mhStep < MH_STEPS; mhStep++) {
                    double u = random.nextDouble() * (pdwSum + wordTable.getProbabilitySum());
                    if(u < pdwSum){
                        int index = lowerBound(pdw, nonzeroTopics.size(), u);
                        newTopic = nonzeroTopics.get(index);
                    }
                    else {
                        if(wordTable.getSampleCount() >= numTopics) {
                            updateAliasTable(word);
                        }
                        newTopic = wordTable.sample(this.random);
                    }

                    if(newTopic != topic) {
                        double newTopicProbability = (matTopicWord[newTopic][word] + beta[word]) / (vecTopic[newTopic] + betaSum);
                        double newFullProbability = (matDocTopic[document][newTopic] + alpha[topic]) * newTopicProbability;
                        double newPdw = matDocTopic[document][newTopic] * newTopicProbability;

                        double oldTopicProbability = (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);
                        double oldFullProbability = (matDocTopic[document][topic] + alpha[topic]) * oldTopicProbability;
                        double oldPdw = matDocTopic[document][topic] * oldTopicProbability;

                        double[] qw = wordTable.getUnnormalizedProbability();
                        double acceptance = (newFullProbability * (oldPdw + qw[topic])) / (oldFullProbability * (newPdw + qw[newTopic]));

                        if(random.nextDouble() < acceptance) {
                            topic = newTopic;
                        }
                    }
                }

                if(matDocTopic[document][topic] == 0){
                    nonzeroDocTopic[document].add(topic);
                }
                incrementCountMatrices(document, word, topic);

                matZ[document][wi] = topic;
            }
        }
    }
}
