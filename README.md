<h3 align="center">AOLME Datasets Generator</h3>
<p align="center">
  Fast CSV dataset generator from transcripts of AOLME Sessions in MS Word format
</p>

## Quick Start

This command line tool has two available options, in the following format:

<pre><code>main.py -i &lt;inputfile> -o &lt;outputfile>
</code></pre>

The generated CSV files will be stored in the <code>output</code> folder, adding a timestamp indicating before the file extension. For example:

<pre><code>$ python main.py -i data/TNS-G-C2L1P-Apr12-C-Issac_q2_01-06.docx -o myoutput</code></pre>

Will generate the CSV file in the output folder:

<pre><code>###################################################################
The CSV file has been created at output/myoutput_20200929165438.csv</code></pre>


