package tu.dortmund.lda.sampler;

import tu.dortmund.lda.data_structure.AliasTable;

import java.util.ArrayList;

public class LightLda extends GibbsLda {

    public static final int MH_STEPS = 2;

    AliasTable[] aliasTables;

    public LightLda(int[][] documents, int vocabularySize, int k) {
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
        for (int document = 0; document < documents.length; document++) {
            double sumPd = documents[document].length + alphaSum;
            for (int wi = 0; wi < documents[document].length; wi++) {
                int word = documents[document][wi];
                int topic = matZ[document][wi];

                decrementCountMatrices(document, word, topic);  

                int oldTopic = topic;
                int newTopic = -1;
                for (int mhStep = 0; mhStep < MH_STEPS; mhStep++) {

                    // Document-Proposal
                    int u = (int)(random.nextDouble() * sumPd);
                    newTopic = u < documents[document].length ? matZ[document][u] : random.nextInt(numTopics);
                    
                    if(topic != newTopic) {
                        double probabilityOfTopic = (matDocTopic[document][topic] + alpha[topic]);
                        double probabilityOfWordInTopic = (matTopicWord[topic][word] + beta[word])
                                / (vecTopic[topic] + betaSum);
                        double proposalTopic = topic == oldTopic ? probabilityOfTopic + 1 : probabilityOfTopic;

                        double probabilityOfNewTopic = matDocTopic[document][newTopic] + alpha[newTopic];
                        double probabilityOfWordInNewTopic = (matTopicWord[newTopic][word] + beta[word])
                                / (vecTopic[newTopic] + betaSum);
                        double proposalNewTopic = newTopic == oldTopic ? probabilityOfNewTopic + 1 : probabilityOfNewTopic;

                        double acceptance = (probabilityOfNewTopic * probabilityOfWordInNewTopic * proposalTopic) 
                        / (probabilityOfTopic * probabilityOfWordInTopic * proposalNewTopic);

                        if(random.nextDouble() < acceptance) {
                            topic = newTopic;
                        }
                    }

                    // Word-Proposal
                    AliasTable wordTable = aliasTables[word];
                    if(wordTable.getSampleCount() >= numTopics) {
                        updateAliasTable(word);
                    }
                    newTopic = wordTable.sample(random);

                    if(topic != newTopic) {
                        double probabilityOfTopic = (matDocTopic[document][topic] + alpha[topic]);
                        double probabilityOfWordInTopic = (matTopicWord[topic][word] + beta[word])
                                / (vecTopic[topic] + betaSum);
                        double proposalTopic = topic == oldTopic
                        ? probabilityOfWordInTopic + (1 / (vecTopic[topic] + betaSum))
                        : probabilityOfWordInTopic;

                        double probabilityOfNewTopic = matDocTopic[document][newTopic] + alpha[newTopic];
                        double probabilityOfWordInNewTopic = (matTopicWord[newTopic][word] + beta[word])
                                / (vecTopic[newTopic] + betaSum);
                        double proposalNewTopic = newTopic == oldTopic
                        ? probabilityOfWordInNewTopic + (1 / (vecTopic[newTopic] + betaSum))
                        : probabilityOfWordInNewTopic;

                        double acceptance = (probabilityOfNewTopic * probabilityOfWordInNewTopic * proposalTopic)
                                / (probabilityOfTopic * probabilityOfWordInTopic * proposalNewTopic);

                        if (random.nextDouble() < acceptance) {
                            topic = newTopic;
                        }
                    }
                }

                incrementCountMatrices(document, word, topic);

                matZ[document][wi] = topic;
            }
        }
    }
}
