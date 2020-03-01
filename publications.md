---
layout: page
sitemap:
  priority: '0.8'
  changefreq: 'always'
permalink: /publications/
title: Publications
---

<!--
Describe your research interests here.
-->

<h2>Publications</h2>
<ul>
	<li>
		<b>Learning from Easy to Complex: Adaptive Multi-curricula Learning for Neural Dialogue Generation. </b><br>
		<i>Hengyi Cai, Hongshen Chen, Cheng Zhang, Yonghao Song, Xiaofang Zhao, Yangxi Li, Dongsheng Duan, Dawei Yin. </i><br>
		In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020), New York, USA.<br>
		<div class="color-button" onclick="isHidden('2020aaai_cai_abstract')">motivation</div>
		<!--<a href="/publications/papers/2020aaai_cai.pdf"><div class="color-button">pdf</div></a> -->
		<!--<div class="color-button" onclick="isHidden('2020aaai_cai_bibtex')">bibtex</div> -->
		<div class="abstract-box" id="2020aaai_cai_abstract" style="display:none">
			<b>Abstract</b>:  Current state-of-the-art neural dialogue systems are mainly data-driven and are trained on human-generated responses. However, due to the subjectivity and open-ended nature of human conversations, the complexity of training dialogues varies greatly.  The noise and uneven complexity of query-response pairs impede the learning efficiency and effects of the neural dialogue generation models.  What is more, so far, there are no unified dialogue complexity measurements, and the dialogue complexity embodies multiple aspects of attributes---specificity, repetitiveness, relevance, etc. Inspired by human behaviors of learning to converse, where children learn from easy dialogues to complex ones and dynamically adjust their learning progress, in this paper, we first analyze five dialogue attributes to measure the dialogue complexity in multiple perspectives on three publicly available corpora. Then, we propose an adaptive multi-curricula learning framework to schedule a committee of the organized curricula. The framework is established upon the reinforcement learning paradigm, which automatically chooses different curricula at the evolving learning process according to the learning status of the neural dialogue generation model. Extensive experiments conducted on five state-of-the-art models demonstrate its learning efficiency and effectiveness with respect to 13 automatic evaluation metrics and human judgments.<br>
			<b>Motivation</b>: <br>
			<ul>
			<li>Training data for neural dialogue models are quite noisy.</li>
			<li>Learn from clean and easy samples first, and then gradually increase the data complexity. (The spirits of curriculum learning)</li>
			<li>Organize the curriculum in terms of multiple empirical attributes---specificity, repetitiveness, relevance, etc. </li>
			</ul>
		</div>
	</li><br>
	<li>
		<b>Posterior-GAN: Towards Informative and Coherent Response Generation with Posterior Generative Adversarial Network.  </b><br>
		<i>Shaoxiong Feng, Hongshen Chen, Kan Li, Dawei Yin. </i><br>
		In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020), New York, USA.<br>
		<div class="color-button" onclick="isHidden('2020aaai_feng_abstract')">motivation</div>
		<a href="/publications/papers/2020aaai_feng.pdf"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2020aaai_feng_bibtex')">bibtex</div>
		<div class="abstract-box" id="2020aaai_feng_abstract" style="display:none">
			<b>Abstract</b>: Neural conversational models learn to generate responses by taking into account the dialog history. These models are typically optimized over the <i>query-response</i> pairs with a maximum likelihood estimation objective. However, the query-response tuples are naturally loosely coupled, and there exist multiple responses that can respond to a given query, which leads the conversational model learning burdensome. Besides, the general dull response problem is even worsened when the model is confronted with meaningless response training instances. Intuitively, a high-quality response not only responds to the given query but also links up to the future conversations, in this paper, we leverage the <i>query-response-future turn</i> triples to induce the generated responses that consider both the given context and the future conversations. To facilitate the modeling of these triples, we further propose a novel encoder-decoder based generative adversarial learning framework, Posterior Generative Adversarial Network (Posterior-GAN), which consists of a forward and a backward generative discriminator to cooperatively encourage the generated response to be informative and coherent by two complementary assessment perspectives. Experimental results demonstrate that our method effectively boosts the informativeness and coherence of the generated response on both automatic and human evaluation, which verifies the advantages of considering two assessment perspectives.<br>
			<b>Motivation</b>: <br>
			<ul>
			<li>A high-quality response not only responds to the given query but also links up to the future conversations.</li>
			<li>Leverage the <i>query-response-future turn</i> triples for training instead of *query-response* pairs. </li>
			<li>Posterior-GAN enables triples training and improves the informativeness and coherence. </li>
			</ul>
		</div>
	</li><br>
	<li>
		<b>Adaptive Parameterization for Neural Dialogue Generation. </b><br>
		<i>Hengyi Cai, Hongshen Chen, Cheng Zhang, Yonghao Song, Xiaofang Zhao and Dawei Yin. </i><br>
		In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019), Hong Kong, China, Nov. 2019.<br>
		<div class="color-button" onclick="isHidden('2019emnlp_cai_abstract')">motivation</div>
		<a href="https://www.aclweb.org/anthology/D19-1188/"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2019emnlp_cai_bibtex')">bibtex</div>
		<a href="https://github.com/hengyicai/AdaND"><div class="color-button">code</div></a>
		<div class="abstract-box" id="2019emnlp_cai_abstract" style="display:none">
			<b>Abstract</b>: Neural conversation systems generate responses based on the sequence-to-sequence (SEQ2SEQ) paradigm. Typically, the model is equipped with a single set of learned parameters to generate responses for given input contexts. When confronting diverse conversations, its adaptability is rather limited and the model is hence prone to generate generic responses. In this work, we propose an Adaptive Neural Dialogue generation model, AdaND, which manages various conversations with conversation-specific parameterization. For each conversation, the model generates parameters of the encoder-decoder by referring to the input context. In particular, we propose two adaptive parameterization mechanisms: a context-aware and a topic-aware parameterization mechanism. The context-aware parameterization directly generates the parameters by capturing local semantics of the given context. The topic-aware parameterization enables parameter sharing among conversations with similar topics by first inferring the latent topics of the given context and then generating the parameters with respect to the distributional topics. Extensive experiments conducted on a large-scale real-world conversational dataset show that our model achieves superior performance in terms of both quantitative metrics and human evaluations.<br>
			<b>Motivation</b>: <br>
			<ul>
			<li>Neural dialogue generation model is prone to generate generic responses when conversations are extremely diverse.</li>
			<li>A single model with diverse parameters manage diverse conversations. </li>
			<li>A context-sensitive local parameterization and a topic-aware global parameterization mechanisms are introduced. </li>
			</ul>
		</div>
		<div class="bibtex-box" id="2019emnlp_cai_bibtex" style="display:none">
			@inproceedings{cai-etal-2019-adaptive, <br>
			&nbsp;&nbsp; title = "Adaptive Parameterization for Neural Dialogue Generation", <br>
			&nbsp;&nbsp; author = "Cai, Hengyi  and Chen, Hongshen  and Zhang, Cheng  and Song, Yonghao  and Zhao, Xiaofang  and Yin, Dawei", <br>
			&nbsp;&nbsp; booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)", <br>
			&nbsp;&nbsp; month = nov, <br>
			&nbsp;&nbsp; year = "2019", <br>
			&nbsp;&nbsp; address = "Hong Kong, China", <br>
			&nbsp;&nbsp; publisher = "Association for Computational Linguistics", <br>
			&nbsp;&nbsp; url = "https://www.aclweb.org/anthology/D19-1188", <br>
			&nbsp;&nbsp; doi = "10.18653/v1/D19-1188", <br>
			&nbsp;&nbsp; pages = "1793--1802" <br>
			}
		</div>
	</li><br>
	<li>
		<b>A Dynamic Product-aware Learning Model for E-commerce Query Intent Understanding.</b><br>
		<i>Jiashu Zhao, Hongshen Chen and Dawei Yin.</i><br>
		In Proceedings of the 28th ACM Conference on Information and Knowledge Management (CIKM 2019), Beijing, China, Oct. 2019.<br>
		<div class="color-button" onclick="isHidden('2019cikm_zhao_abstract')">motivation</div>
		<a href="/publications/papers/2019cikm_zhao.pdf"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2019cikm_zhao_bibtex')">bibtex</div>
		<div class="abstract-box" id="2019cikm_zhao_abstract" style="display:none">
			<b>Abstract</b>: Query intent understanding is a fundamental and essential task in searching, which promotes personalized retrieval results and users' satisfaction. In E-commerce, query understanding is particularly referring to bridging the gap between query representations and product representations. In this paper, we aim to map the queries into the predefined tens of thousands of fine-grained categories extracted from the product descriptions. The problem is very challenging in several aspects. First, a query may be related to multiple categories and to identify all the best matching categories could eventually drive the search engine for high recall and diversity. Second, the same query may have dynamic intents under various scenarios and there is a need to distinguish the differences to promote accurate categories of products. Third, the tail queries are particularly difficult for understanding due to noise and lack of customer feedback information. To better understand the queries, we firstly conduct analysis on the search queries and behaviors in the E-commerce domain and identified the uniqueness of our problem (e.g. longer sessions). Then we propose a <i>D</i>ynamic <i>P</i>roduct-aware <i>H</i>ierarchical <i>A</i>ttention (<i>DPHA</i>) framework to capture the explicit and implied meanings of a query given its context information in the session. Specifically, <i>DPHA</i> automatically learns the bidirectional query-level and self-attentional session-level representations which can capture both complex long range dependencies and structural information. Extensive experimental results on a real E-commerce query data set demonstrate the effectiveness of the proposed <i>DPHA</i> compared to the state-of-art baselines. <br>
			<b>Motivation</b>: <br>
			<ul>
			<li>Understand query intent through session-level representation with self-attention mechanism.</li>
			<li>Illustrate query-intent distributions. </li>
			</ul>
		</div>
		<div class="bibtex-box" id="2019cikm_zhao_bibtex" style="display:none">
		@inproceedings{zhao2019dynamic, <br>
		&nbsp;&nbsp; title={A Dynamic Product-aware Learning Model for E-commerce Query Intent Understanding}, <br>
		&nbsp;&nbsp; author={Zhao, Jiashu and Chen, Hongshen and Yin, Dawei}, <br>
		&nbsp;&nbsp; booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management}, <br>
		&nbsp;&nbsp; pages={1843--1852}, <br>
		&nbsp;&nbsp; year={2019}, <br>
		&nbsp;&nbsp; organization={ACM} <br>
		}
		</div>
	</li><br>
	<li>
		<b>Fine-Grained Product Categorization in E-commerce.</b><br>
		<i>Hongshen Chen, Jiashu Zhao and Dawei Yin. </i><br>
		In Proceedings of the 28th ACM Conference on Information and Knowledge Management (CIKM 2019), Beijing, China, Oct. 2019.<br>
		<div class="color-button" onclick="isHidden('2019cikm_chen_abstract')">motivation</div>
		<a href="/publications/papers/2019cikm_chen.pdf"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2019cikm_chen_bibtex')">bibtex</div>
		<div class="abstract-box" id="2019cikm_chen_abstract" style="display:none">
			<b>Abstract</b>: E-commerce sites usually leverage taxonomies for better organizing products. The fine-grained categories, regarding the leaf categories in taxonomies, are defined by the most descriptive and specific words of products. Fine-grained product categorization remains challenging, due to blurred concepts of fine grained categories (i.e. multiple equivalent or synonymous categories), instable category vocabulary (i.e. the emerging new products and the evolving language habits), and lack of labelled data. To address these issues, we proposes a novel <b>N</b>eural <b>P</b>roduct <b>C</b>ategorization model---NPC to identify fine-grained categories from the product content. NPC is equipped with a character-level convolutional embedding layer to learn the compositional word representations, and a spiral residual layer to extract the word context annotations capturing complex long range dependencies and structural information. To perform categorization beyond predefined categories, NPC categorizes a product by jointly recognizing categories from the product content and predicting categories from predefined category vocabularies. Furthermore, to avoid extensive human labors, NPC is able to adapt to weak labels, generated by mining the search logs,  where the customers' behaviors naturally connect products with categories. Extensive experiments performed on a real e-commerce platform datasets illustrate the effectiveness of the proposed models.<br>
			<b>Motivation</b>: <br>
			<ul>
			<li>Product categories can be recognized from produc contents and classified from product category vocabulary.</li>
			<li>Instead of a manual labelling corpus, large scale corpus with weak labels can be mined from search logs. </li>
			</ul>
		</div>
		<div class="bibtex-box" id="2019cikm_chen_bibtex" style="display:none">
		@inproceedings{chen2019fine, <br>
		&nbsp;&nbsp; title={Fine-Grained Product Categorization in E-commerce}, <br>
		&nbsp;&nbsp; author={Chen, Hongshen and Zhao, Jiashu and Yin, Dawei}, <br>
		&nbsp;&nbsp; booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management}, <br>
		&nbsp;&nbsp; pages={2349--2352}, <br>
		&nbsp;&nbsp; year={2019}, <br>
		&nbsp;&nbsp; organization={ACM} <br>
		}
		</div>
	</li><br>
	<li>
		<b>Explicit State Tracking with Semi-Supervision for Neural Dialogue Generation.</b><br>
		<i>Xisen Jin, Wenqiang Lei, Zhaochun Ren, Hongshen Chen, Shangsong Liang, Yihong Eric Zhao, Dawei Yin.</i><br>
		In Proceedings of the 27th ACM Conference on Information and Knowledge Management (CIKM 2018), Turin, Italy, Oct. 2018.<br>
		<a href="https://arxiv.org/abs/1808.10596"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2018cikm_jin_bibtex')">bibtex</div>
		<div class="bibtex-box" id="2018cikm_jin_bibtex" style="display:none">
		@inproceedings{jin2018explicit, <br>
		&nbsp;&nbsp; title={Explicit State Tracking with Semi-Supervisionfor Neural Dialogue Generation}, <br>
		&nbsp;&nbsp; author={Jin, Xisen and Lei, Wenqiang and Ren, Zhaochun and Chen, Hongshen and Liang, Shangsong and Zhao, Yihong and Yin, Dawei}, <br>
		&nbsp;&nbsp; booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management}, <br>
		&nbsp;&nbsp; pages={1403--1412}, <br>
		&nbsp;&nbsp; year={2018}, <br>
		&nbsp;&nbsp; organization={ACM} <br>
		}
		</div>
	</li><br>
	<li>
		<b>Knowledge Diffusion for Neural Dialogue Generation.</b><br>
		<i>Liu Shuman, Hongshen Chen, Zhaochun Ren, Yang Feng, Qun Liu and Dawei Yin.</i><br>
		ACL 2018, Melbourne, Australia, 2018.<br>
		<a href="https://www.aclweb.org/anthology/P18-1138/"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2018acl_liu_bibtex')">bibtex</div>
		<a href="https://github.com/liushuman/neural-knowledge-diffusion"><div class="color-button">corpus</div></a>
		<div class="bibtex-box" id="2018acl_liu_bibtex" style="display:none">
		@inproceedings{liu-etal-2018-knowledge,
		&nbsp;&nbsp; title = "Knowledge Diffusion for Neural Dialogue Generation", <br>
		&nbsp;&nbsp; author = "Liu, Shuman  and Chen, Hongshen  and Ren, Zhaochun  and Feng, Yang  and Liu, Qun  and Yin, Dawei", <br>
		&nbsp;&nbsp; booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)", <br>
		&nbsp;&nbsp; month = jul, <br>
		&nbsp;&nbsp; year = "2018", <br>
		&nbsp;&nbsp; address = "Melbourne, Australia", <br>
		&nbsp;&nbsp; publisher = "Association for Computational Linguistics", <br>
		&nbsp;&nbsp; url = "https://www.aclweb.org/anthology/P18-1138", <br>
		&nbsp;&nbsp; doi = "10.18653/v1/P18-1138", <br>
		&nbsp;&nbsp; pages = "1489--1498", <br>
		&nbsp;&nbsp; abstract = "End-to-end neural dialogue generation has shown promising results recently, but it does not employ knowledge to guide the generation and hence tends to generate short, general, and meaningless responses. In this paper, we propose a neural knowledge diffusion (NKD) model to introduce knowledge into dialogue generation. This method can not only match the relevant facts for the input utterance but diffuse them to similar entities. With the help of facts matching and entity diffusion, the neural dialogue generation is augmented with the ability of convergent and divergent thinking over the knowledge base. Our empirical study on a real-world dataset prove that our model is capable of generating meaningful, diverse and natural responses for both factoid-questions and knowledge grounded chi-chats. The experiment results also show that our model outperforms competitive baseline models significantly." <br>
		}
		</div>
	</li><br>
	<li>
		<b>Learning Tag Dependencies for Sequence Tagging.</b><br>
		<i>Yuan Zhang, Hongshen Chen, Yihong Eric Zhao, Qun Liu, Dawei Yin.</i><br>
		IJCAI, 2018.<br>
		<a href="https://www.ijcai.org/proceedings/2018/0637"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2018ijcai_zhang_bibtex')">bibtex</div>
		<div class="bibtex-box" id="2018ijcai_zhang_bibtex" style="display:none">
		@inproceedings{ijcai2018-0637, <br>
		&nbsp;&nbsp; title     = {Learning Tag Dependencies for Sequence Tagging}, <br>
		&nbsp;&nbsp; author    = {Yuan Zhang and Hongshen Chen and Yihong Zhao and Qun Liu and Dawei Yin}, <br>
		&nbsp;&nbsp; booktitle = {Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, {IJCAI-18}}, <br>
		&nbsp;&nbsp; publisher = {International Joint Conferences on Artificial Intelligence Organization}, <br>
		&nbsp;&nbsp; pages     = {4581--4587}, <br>
		&nbsp;&nbsp; year      = {2018}, <br>
		&nbsp;&nbsp; month     = {7}, <br>
		&nbsp;&nbsp; doi       = {10.24963/ijcai.2018/637}, <br>
		&nbsp;&nbsp; url       = {https://doi.org/10.24963/ijcai.2018/637} <br>
		}
		</div>
	</li><br>
	<li>
		<b>Hierarchical Variational Memory Network for Dialogue Generation. </b><br>
		<i>Hongshen Chen, Zhaochun Ren, Jiliang Tang, Yihong Eric Zhao and Dawei Yin.</i><br>
		WWW,2018.<br>
		<a href="/publications/papers/2018www.pdf"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2018www_bibtex')">bibtex</div>
		<a href="https://github.com/chenhongshen/HVMN"><div class="color-button">code&corpus</div></a>
		<div class="bibtex-box" id="2018www_bibtex" style="display:none">
		@inproceedings{chen2018hierarchical, <br>
		&nbsp;&nbsp; title={Hierarchical variational memory network for dialogue generation}, <br>
		&nbsp;&nbsp; author={Chen, Hongshen and Ren, Zhaochun and Tang, Jiliang and Zhao, Yihong Eric and Yin, Dawei}, <br>
		&nbsp;&nbsp; booktitle={Proceedings of the 2018 World Wide Web Conference}, <br>
		&nbsp;&nbsp; pages={1653--1662}, <br>
		&nbsp;&nbsp; year={2018}, <br>
		&nbsp;&nbsp; organization={International World Wide Web Conferences Steering Committee} <br>
		}
		</div>
	</li><br>
	<li>
		<b>A Survey on Dialogue Systems: Recent Advances and New Frontiers.</b><br>
		<i>Hongshen Chen, Xiaorui Liu, Dawei Yin and Jiliang Tang. </i><br>
		SIGKDD Explorations, 2018.<br>
		<a href="https://arxiv.org/abs/1711.01731"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2018kdd_exp_bibtex')">bibtex</div>
		<div class="bibtex-box" id="2018kdd_exp_bibtex" style="display:none">
		@article{chen2017survey, <br>
		&nbsp;&nbsp; title={A survey on dialogue systems: Recent advances and new frontiers}, <br>
		&nbsp;&nbsp; author={Chen, Hongshen and Liu, Xiaorui and Yin, Dawei and Tang, Jiliang}, <br>
		&nbsp;&nbsp; journal={Acm Sigkdd Explorations Newsletter}, <br>
		&nbsp;&nbsp; volume={19}, <br>
		&nbsp;&nbsp; number={2}, <br>
		&nbsp;&nbsp; pages={25--35}, <br>
		&nbsp;&nbsp; year={2017}, <br>
		&nbsp;&nbsp; publisher={ACM} <br>
		}
		</div>
	</li><br>
	<li>
		<b>Learning dependency edge transfer rule representation using encoder-decoder (in Chinese).</b><br>
		<i>Hongshen Chen, Qun Liu.</i><br>
		Scientia Sinica Informationis, 2017.<br>
		<!--
		<a href=""><div class="color-button">pdf</div></a><a href=""><div class="color-button">cite</div></a><a href=""><div class="color-button">code</div></a>
		-->
	</li><br>
	<li>
		<b>Neural Network for Heterogeneous Annotations.</b><br>
		<i>Hongshen Chen, Yue Zhang, Qun Liu.</i><br>
		EMNLP, 2016.<br>
		<a href="https://www.aclweb.org/anthology/D16-1070/"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2016emnlp_bibtex')">bibtex</div>
		<a href="https://github.com/chenhongshen/NNHetSeq"><div class="color-button">code&corpus</div></a>
		<div class="bibtex-box" id="2016emnlp_bibtex" style="display:none">
		@inproceedings{chen-etal-2016-neural, <br>
		&nbsp;&nbsp; title = "Neural Network for Heterogeneous Annotations", <br>
		&nbsp;&nbsp; author = "Chen, Hongshen  and <br>
		&nbsp;&nbsp; Zhang, Yue  and <br>
		&nbsp;&nbsp; Liu, Qun", <br>
		&nbsp;&nbsp; booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing", <br>
		&nbsp;&nbsp; month = nov, <br>
		&nbsp;&nbsp; year = "2016", <br>
		&nbsp;&nbsp; address = "Austin, Texas", <br>
		&nbsp;&nbsp; publisher = "Association for Computational Linguistics", <br>
		&nbsp;&nbsp; url = "https://www.aclweb.org/anthology/D16-1070", <br>
		&nbsp;&nbsp; doi = "10.18653/v1/D16-1070", <br>
		&nbsp;&nbsp; pages = "731--741", <br>
		}
		</div>
	</li><br>
	<li>
		<b>A Dependency Edge Transfer Translation Rule Generator.</b><br>
		<i>Hongshen Chen, Qun Liu. </i><br>
		CCL, 2016.<br>
		<!--
		<a href=""><div class="color-button">pdf</div></a><a href=""><div class="color-button">cite</div></a><a href=""><div class="color-button">code</div></a>
		-->
	</li><br>
	<li>
		<b>A Dependency Edge-based Transfer Model for Statistical Machine Translation. </b><br>
		<i>Hongshen Chen, Jun Xie, Fandong Meng, Weibin Jiang, Qun Liu. </i><br>
		COLING, 2014.<br>
		<a href="https://www.aclweb.org/anthology/C14-1104/"><div class="color-button">pdf</div></a>
		<div class="color-button" onclick="isHidden('2014coling_bibtex')">bibtex</div>
		<div class="bibtex-box" id="2014coling_bibtex" style="display:none">
		@inproceedings{chen-etal-2014-dependency, <br>
		&nbsp;&nbsp; title = "A Dependency Edge-based Transfer Model for Statistical Machine Translation", <br>
		&nbsp;&nbsp; author = "Chen, Hongshen  and <br>
		&nbsp;&nbsp; Xie, Jun  and <br>
		&nbsp;&nbsp; Meng, Fandong  and <br>
		&nbsp;&nbsp; Jiang, Wenbin  and <br>
		&nbsp;&nbsp; Liu, Qun", <br>
		&nbsp;&nbsp; booktitle = "Proceedings of {COLING} 2014, the 25th International Conference on Computational Linguistics: Technical Papers", <br>
		&nbsp;&nbsp; month = aug, <br>
		&nbsp;&nbsp; year = "2014", <br>
		&nbsp;&nbsp; address = "Dublin, Ireland", <br>
		&nbsp;&nbsp; publisher = "Dublin City University and Association for Computational Linguistics", <br>
		&nbsp;&nbsp; url = "https://www.aclweb.org/anthology/C14-1104", <br>
		&nbsp;&nbsp; pages = "1103--1113", <br>
		}
		</div>
	</li><br>
</ul>

<!--
<h2>Research Projects</h2>
<ul>
	<li>
		<b>Project title</b><br>
		University, Duration<br>
		<i>Other details such as advisor's name may go here</i><br>
		<a href=""><div class="color-button">report</div></a><a href=""><div class="color-button">code</div></a>
	</li><br>
	<li>
		<b>Project title</b><br>
		University, Duration<br>
		<i>Other details such as advisor's name may go here</i><br>
		<a href=""><div class="color-button">report</div></a><a href=""><div class="color-button">code</div></a>
	</li><br>
</ul>

<h2>Research Implementations</h2>
<ul>
	<li>
		<b>Title #1</b>: Brief description of this research implementation.<br>
		<a href=""><div class="color-button">paper</div></a><a href=""><div class="color-button">report</div></a><a href=""><div class="color-button">code</div></a>
	</li><br>
	<li>
		<b>Title #2</b>: Brief description of this research implementation.<br>
		<a href=""><div class="color-button">paper</div></a><a href=""><div class="color-button">report</div></a><a href=""><div class="color-button">code</div></a>
	</li><br>
</ul>
-->
