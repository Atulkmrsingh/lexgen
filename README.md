## LexGen: Domain-aware Multilingual Lexicon Generation</h1>
> [LexGen: Domain-aware Multilingual Lexicon Generation](https://arxiv.org/abs/2405.11200)  
> Ayush Maheshwari*, Atul Kumar Singh*, Karthika NJ, Krishnakant Bhatt, Preethi Jyothi, Ganesh Ramakrishnan      
> Association for Computational Linguistics (__ACL Main__) 2025
<p>
LexGen is a domain-aware lexicon generation framework that builds specialized multilingual dictionaries from English to six Indic languages — Hindi, Kannada, Gujarati, Marathi, Odia, and Tamil — across multiple technical and non-technical domains. It introduces a novel <strong>Domain Routing (DR)</strong> layer in a Transformer-based architecture to selectively route information through domain-specific or shared pathways, enabling both in-domain learning and generalization to unseen domains or languages.
</p>

<h2>✨ Highlights</h2>
<ul>
  <li><strong>Multilingual & Multi-domain</strong>: Trained on over 75K expert-curated translation pairs across 6 Indian languages and 8 domains.</li>
  <li><strong>Domain Routing Layer</strong>: Dynamically gates tokens through domain-specific or shared decoder paths.</li>
  <li><strong>Zero-shot and Few-shot Capable</strong>: Generalizes to new domains and languages even with limited supervision.</li>
  <li><strong>Benchmark Dataset</strong>: Created using dictionaries from the Commission for Scientific and Technical Terminology (CSTT), India.</li>
</ul>

<h2>🛠️ Setup</h2>

<h3>1. Clone the pre-trained base model</h3>
<p>Download and extract the base multilingual model used for fine-tuning:</p>

<pre><code>wget https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/en2indic.zip
unzip en2indic.zip
</code></pre>
<p>Place the extracted contents in the root directory of this project.</p>

<h3>2. Prepare the Environment</h3>
<p>Set up your Python environment, install dependencies, and ensure <code>fairseq</code> is correctly installed.</p>

<h2>🔧 Code Modifications (Required before training)</h2>

<p>To enable domain-aware routing and correct training behavior, perform the following changes:</p>

<h4>Modify <code>fairseq/fairseq/tasks/translation.py</code></h4>
<ul>
  <li>Comment line <strong>373</strong></li>
  <li>Uncomment line <strong>375</strong></li>
</ul>

<h4>Modify <code>fairseq/fairseq/data/language_pair_dataset.py</code></h4>
<ul>
  <li>Comment lines: <strong>217</strong>, <strong>251</strong>, <strong>364</strong></li>
  <li>Uncomment line: <strong>363</strong></li>
</ul>

<h2>📂 Data</h2>
<p>All training and evaluation data is stored inside the <code>data/</code> folder. This includes parallel lexicon pairs categorized by language and domain.</p>

<h2>🚀 Training</h2>
<p>To start training the LexGen model, run:</p>

<pre><code>bash traincode.sh
</code></pre>

<p>This script performs full fine-tuning of the LexGen model using the base multilingual checkpoint and curated lexicon pairs.</p>

<h2>🔎 Inference & Benchmarking</h2>
<p>To run inference on test data or evaluate the model across domains and languages:</p>

<pre><code>bash run_benchmark.sh
</code></pre>

<h2>📄 Citation</h2>
<p>If you use LexGen or its dataset in your research, please cite the corresponding ACL submission.</p>


```bibtex
@inproceedings{maheshwari-etal-2025-lexgen,
    title = "LexGen: Domain-aware Multilingual Lexicon Generation",
    author = "Maheshwari, Ayush and Singh, Atul Kumar and NJ, Karthika and Bhatt, Krishnakant and Jyothi, Preethi and Ramakrishnan, Ganesh",
    booktitle = "Proceedings of the 63rd Conference of the Association for Computational Linguistics: Main Volume",
    year = "2025",
    publisher = "Association for Computational Linguistics",
}
```

<h2>📜 License</h2>
<p>This project is for academic research and benchmarking purposes only.</p>
