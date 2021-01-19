# LDA Samplers

This repository includes several **Latent Dirichlet Allocation (LDA)** samplers implemented in **Java**.
The code should help to understand the different sampling algorithms.
It therefore sometimes favours readability over efficiency. 
All algorithms are single thread and run only in main memory.

## How to run the project?
This project utilizes **Apache Ant** for automating the build process. 
Simply install **Apache Ant** and **Java** on your system.
You can now run the following commands in the project directory:

* `ant build` to build the project (jar file),
* `ant run` to run the main method which executes a small sample,
* `ant doc` to generate a documentation,
* `ant clean` to clean up project folder.

## Which samlers are included?

The implemented LDA samplers are:
* The standart **Collapsed Gibbs Sampler** [[Gibbs]](#gibbs) which has a O( log *T* ) complexity per word token.
* The **Sparse LDA** [[Sparse]](#sparse) algorithm, which has a O( *Td* + *Tv* ) complexity. 
* The **Alias LDA** [[Alias]](#alias) algorithm, which has a O( *Td* ) sampling complexity.
* The **FTree LDA** [[FTree]](#ftree) algorithm, which has a O( *Td* ) complexity.
* The **Light LDA** [[Light]](#light) algorithm, which has a O( 1 ) complexity. 

Here *Td* represents the number of actually instantiated topics in document *d*, while *Tv* represents the number of topics occurring for a particular term *v*.
Another known sampler is **WrapLDA** [[Wrap]](#light), which uses the exact same sampling method as **LightLDA** but in a more optimized way, such that it greatly reduces random memory access. Since the, in this repository, implemented algorithms run only in main memory, **WrapLDA** and **LightLDA** can be seen as the same algorithm.

### Discussion:
The paper [[Comparison]](#comparison) gives a comparison between the sampling processes of AliasLDA, FTreeLDA, LightLDA, and WarpLDA. These are four of the currently best performing samplers, with emphasis on analyzing huge corpora with possibly millions of documents and thousands of topics. Each one has its own advantages and disadvantages. However, in practice it seems that FTree LDA and WrapLDA perform the best on large datasets. The Metropolis Hastings based WarpLDA incurs an O( 1 ) sampling complexity per word. However since it relies on proposals with acceptance rate π, only 1 / π of the proposals get accepted. Furthermore, constructing an alias table is more expensive than constructing an F+Tree. FTree LDA on the other hand has an O( *Td* ) sampling complexity per word, but no rejection rate.
Which of the both samplers performs best is therefore mostly dependent on the average document length. On a dataset with long documents, the Metropolis Hastings based WarpLDA is faster. In most other scenarios, FTree LDA will be faster.

This does however not mean that the normal Gibbs Sampler and Sparse LDA are useless. 
In a scenario where one analyses a relatively small datasets with a moderate number of topics on a single machine, Sparse LDA may significantly outperform Wrap LDA and FTree LDA as it has less overhead and does not rely on expensive data structures to amortize complexity.
On a very small dataset with just a few topics, the normal Gibbs Sampler may even be the fastest algorithm.

When analyzing very large corpora that span over millions of documents, the discussed algorithms usually do not fit into main memory. They therefore have to rely on storage media to access corpora information and count matrices. To further accelerate the sampling process, the workload is usually spread across multiple machines. Modern implementations often utilize a so called Parameter Server Framework which allows them to distribute the workload to multiple machines while maintaining globally shared parameters. At a large scale, performance of a sampler is therefore also dependent on how data is accessed and how well it can be parallelized on multiple machines. 

## References
<a id="gibbs">[Gibbs]</a> Thomas L Griffiths and Mark Steyvers. Finding scientific topics. 
*Proceedings of the National academy of Sciences*, 101(1):5228–5235, 2004.

<a id="sparse">[Sparse]</a> Limin Yao, David Mimno, and Andrew McCallum. Efficient methods for topic model inference on streaming document collections. 
In *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, page 937–946, 2009.

<a id="alias">[Alias]</a> Aaron Q. Li, Amr Ahmed, Sujith Ravi, and Alexander J. Smola. Reducing the sampling complexity of topic models. 
In *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, page 891–900, 2014.

<a id="light">[Light]</a> Jinhui Yuan, Fei Gao, Qirong Ho, Wei Dai, Jinliang Wei, Xun Zheng, Eric Po Xing, Tie-Yan Liu, and Wei-Ying Ma. Lightlda: Big topic models on modest computer clusters. 
In *Proceedings of the 24th International Conference on World Wide Web*, page 1351–1361, 2015.

<a id="ftree">[FTree]</a> Hsiang-Fu Yu, Cho-Jui Hsieh, Hyokun Yun, S.V.N. Vishwanathan, and Inderjit S. Dhillon. A scalable asynchronous distributed algorithm for topic modeling. 
In *Proceedings of the 24th International Conference on World Wide Web*, page 1340–1350, 2015.

<a id="wrap">[Wrap]</a> Jianfei Chen, Kaiwei Li, Jun Zhu, and Wenguang Chen. Warplda: A cache efficient o(1) algorithm for latent dirichlet allocation. 
*Proceedings of the VLDB Endowment*, 9(1):744–755, 2016.

<a id="comparison">[Comparison]</a> [48] Lele Yut, Ce Zhang, Yingxia Shao, and Bin Cui. Lda*: A robust and large-scale topic modeling system. 
*Proceedings of the VLDB Endowment*, 10(11):1406–1417, 2017.
