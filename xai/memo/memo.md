# メモ

## やるべきこと
- 論文を読んで、それを理解すること
## わからないこと
- Affin
## わかったこと
- 現状
    - 問題設定として、化合物の性質予測（一つの化合物のデータからその化合物の性質を予測する問題）と、二つのものの反応予測（二つの化合物の反応、化合物とタンパク質の反応、タンパク質同士の反応）がおもにある。
    - 化合物の解釈性はLIME、SHAP（GradCAMは見てない）の方法に関する研究がある
    - タンパク質の予測は、アミノ酸の配列を自然言語のようにして、エンべディングするので、RNNの解釈性は、ほとんどない。でも、Attention機構を用いて、エンべディングする手法を採択すれば、解釈することは可能。
- どんな問題設定とどのように解くか?
    - 
- 重要ワード
    - アッセイとは、検体の存在、量、または機能的な活性や反応を、定性的に評価、または定量的に測定する方法
    - QSAR()