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

                    // Doc-Proposal
                    int u = (int)(random.nextDouble() * sumPd);
                    newTopic = u < documents[document].length ? matZ[document][u] : random.nextInt(numTopics);
                    
                    if(topic != newTopic) {
                        double tmpOld = (matDocTopic[document][topic] + alpha[topic])
                                * (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);
                        double propOld = matDocTopic[document][topic] + alpha[topic];
                        if(topic == oldTopic) propOld += 1;

                        double tmpNew = (matDocTopic[document][newTopic] + alpha[newTopic])
                                * (matTopicWord[newTopic][word] + beta[word]) / (vecTopic[newTopic] + betaSum);
                        double propNew = matDocTopic[document][newTopic] + alpha[newTopic];
                        if(newTopic == oldTopic) propNew += 1;

                        double acceptance = (tmpNew * propOld) / (tmpOld * propNew);

                        if(random.nextDouble() < acceptance) {
                            topic = newTopic;
                        }



                        // double acceptance = 
                        //         (matDocTopic[document][newTopic] + alpha[newTopic]) 
                        //         * (matTopicWord[newTopic][word] + beta[word])
                        //         * (vecTopic[topic] + betaSum)
                        //         * (alpha[topic] + topic == oldTopic ? matDocTopic[document][topic] + 1 : matDocTopic[document][topic])
                        //         / 
                        //         (matDocTopic[document][topic] + alpha[topic]) 
                        //         * (matTopicWord[topic][word] + beta[word])
                        //         * (vecTopic[newTopic] + betaSum)
                        //         * (alpha[newTopic] + topic == newTopic ? matDocTopic[document][newTopic] + 1 : matDocTopic[document][newTopic]);

                        // if (random.nextDouble() < acceptance) {
                        //     topic = newTopic;
                        // }



                        // double propOfTopic = (matDocTopic[document][topic] + alpha[topic]);
                        // double propOfWordInTopic = (matTopicWord[topic][word] + beta[word])
                        //         / (vecTopic[topic] + betaSum);

                        // double propOfNewTopic = matDocTopic[document][newTopic] + alpha[newTopic];
                        // double propOfWordInNewTopic = (matTopicWord[newTopic][word] + beta[word])
                        //         / (vecTopic[newTopic] + betaSum);

                        // double objectiveFunction = propOfNewTopic * propOfWordInNewTopic / propOfTopic * propOfWordInTopic;

                        // double proposalTopic = topic == oldTopic ? propOfTopic + 1 : propOfTopic;
                        // double proposalNewTopic = newTopic == oldTopic ? propOfNewTopic + 1 : propOfNewTopic;

                        // double acceptance = objectiveFunction * proposalTopic / proposalNewTopic;

                        // if(random.nextDouble() < acceptance) {
                        //     topic = newTopic;
                        // }
                    }

                    // Word-Proposal
                    AliasTable wordTable = aliasTables[word];
                    if(wordTable.getSampleCount() >= numTopics) {
                        updateAliasTable(word);
                    }
                    newTopic = wordTable.sample(random);

                    if(topic != newTopic) {
                        double tmpOld = (matDocTopic[document][topic] + alpha[topic])
                                * (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);
                        double tmpNew = (matDocTopic[document][newTopic] + alpha[newTopic])
                                * (matTopicWord[newTopic][word] + beta[word]) / (vecTopic[newTopic] + betaSum);

                        double acceptance = tmpNew * (wordTable.getUnnormalizedProbability()[topic] / alpha[topic])
                                / tmpOld * (wordTable.getUnnormalizedProbability()[newTopic] / alpha[newTopic]);

                        if(random.nextDouble() < acceptance) {
                            topic = newTopic;
                        }





                        // double tmpOld = (matDocTopic[document][topic] + alpha[topic])
                        //     * (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);
                        // double tmpNew = (matDocTopic[document][newTopic] + alpha[newTopic])
                        //     * (matTopicWord[newTopic][word] + beta[word]) / (vecTopic[newTopic] + betaSum);

                        // double proposalTopic = topic == oldTopic
                        // ? (matTopicWord[topic][word] + 1 + beta[word]) / (vecTopic[topic] + betaSum)
                        // : (matTopicWord[topic][word] + beta[word]) / (vecTopic[topic] + betaSum);

                        // double proposalNewTopic = newTopic == oldTopic
                        // ? (matTopicWord[newTopic][word] + 1 + beta[word]) / (vecTopic[newTopic] + betaSum)
                        // : (matTopicWord[newTopic][word] + beta[word]) / (vecTopic[newTopic] + betaSum);

                        // double acceptance = tmpNew * proposalTopic
                        // / tmpOld * proposalNewTopic;

                        // if(random.nextDouble() < acceptance) {
                        // topic = newTopic;
                        // }





                        // double propOfTopic = matDocTopic[document][topic] + alpha[topic];
                        // double propOfWordInTopic = (matTopicWord[topic][word] + beta[word])
                        //         / (vecTopic[topic] + betaSum);

                        // double propOfNewTopic = matDocTopic[document][newTopic] + alpha[newTopic];
                        // double propOfWordInNewTopic = (matTopicWord[newTopic][word] + beta[word])
                        //         / (vecTopic[newTopic] + betaSum);

                        // double objectiveFunction = propOfNewTopic * propOfWordInNewTopic / propOfTopic
                        //         * propOfWordInTopic;
                        
                        // double proposalTopic = topic == oldTopic
                        //         ? (matTopicWord[topic][word] + 1 + beta[word]) / (vecTopic[topic] + betaSum)
                        //         : propOfTopic;

                        // double proposalNewTopic = newTopic == oldTopic 
                        //         ? (matTopicWord[newTopic][word] + 1 + beta[word]) / (vecTopic[newTopic] + betaSum)
                        //         : propOfNewTopic;

                        // double acceptance = objectiveFunction * proposalTopic / proposalNewTopic;

                        // if (random.nextDouble() < acceptance) {
                        //     topic = newTopic;
                        // }
                    }
                }

                incrementCountMatrices(document, word, topic);

                matZ[document][wi] = topic;
            }
        }
    }
}
