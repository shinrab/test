## LaTeXのコンパイルについて

LaTeXの作成環境によって，利用できるコンパイラ，パッケージ，フォント，設定画面などには多少の違いがあります．  
ただし，使用するコンパイル方式や必要なファイルを揃えれば，基本的にはどの環境でも同等の出力結果を得られます．

本リポジトリでは，VS Code，Overleaf，CloudLaTeXの各環境に対応したファイルを用意しています．
コンパイル方式や環境ごとの設定について詳しく知りたい場合は，`LaTeX_explanation.pdf`を参照してください．

各自で次のいずれかの環境を確認し，正常にコンパイルできるか確認してください．

## 手順

1. GitHubからこのリポジトリをZIP形式でダウンロードする．

2. 使用する環境に応じて，プロジェクトを開く．

   - **VS Code（TeX Live）**
     - ZIPファイルを展開する．
     - 展開したフォルダをVS Codeで開く．

   - **Overleaf**
     - `New Project` → `Upload Project`を選択する．
     - ダウンロードしたZIPファイルをアップロードする．

   - **CloudLaTeX**
     - 新しいプロジェクトを作成する．
     - ZIPファイル，または展開したプロジェクト内のファイルをアップロードする．

3. 練習用ファイルである`LaTeX_practice.tex`を開く．

4. コンパイラを設定する．すべての環境で同じコンパイル方式を使用する．  
   各環境の設定画面やビルド設定から，以下のように選択すること．

   - **VS Code（TeX Live）**
     - LaTeX Workshopなどのビルド設定で`LuaLaTeX`を選択する．

   - **Overleaf**
     - `File` → `Settings` → `Compiler`から`LaTeX`を選択する．
     - さらに，`latexmkrc`が`LaTeX_practice.tex`と同じディレクトリーに入っているのか，`latexmkrc`に拡張子が付いてないか，必ず確認する．
   
   - **CloudLaTeX**
     - プロジェクトのコンパイラ設定から`LuaLaTeX`を選択する．

5. `LaTeX_practice.tex`をコンパイルする．

6. 同じ環境で，解答用ファイルである`LaTeX_practice_answer.tex`もコンパイルする．

7. コンパイルに問題がないことが確認出来たら，`LaTeX_practice.tex`を修正し，課題に取り込む．

8. `homework`というフォルダーの中に課題を提出する．この際のファイルの名前は`homework/LaTeX_practice_{自分の名前}.tex`にする．
