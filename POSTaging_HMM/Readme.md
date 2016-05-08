This is a POS tagger based on a HMM implementation. It uses the METU-Sabanci Turkish Dependency Treebank dataset.

In order to train the HMM tagger use the following command
<pre><code> python train_hmm_tagger.py training_filename 􀀀􀀀[--cpostag|--postag] </code></pre>

In order to tag a file use the following command
<pre><code> python hmm_tagger.py test_blind_filename output_filename </code></pre>

In order to evaluate the performace of the tagger use the following command
<pre><code> python evaluate_hmm_tagger.py output_filename test_gold_filename </code></pre>