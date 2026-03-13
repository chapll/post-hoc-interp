\section{Approach}

Our pipeline consists of four stages: (1) constructing a sleeper model organism that exhibits controllable post-hoc reasoning, (2) extracting a linear direction from activation differences between the sleeper model organism's two reasoning modes, (3) causally validating the direction via activation steering, and (4) testing cross-domain transfer to the unmodified base model. All experiments use Qwen3-8B \cite{yang2025qwen3technicalreport} (36 layers, 4096 hidden dim, bfloat16). The sleeper model organism concept is adapted from \citet{hubinger2024sleeper}, who used sleeper agents to study deceptive alignment; we repurpose the paradigm to create a controlled source of post-hoc reasoning. The verification format, force-answer probing, multi-trigger deconfounding, direction extraction pipeline, and the full causal steering and transfer evaluation framework are our original contributions. Causal steering uses the activation addition method of \citet{turner2023activation}.

\subsection{Verification Format}

We convert all problems into a binary forced-choice format with two answer tokens: \verb|"Mu"| and \verb|"Nu"|, both single tokens in the Qwen3 vocabulary. The mapping between \verb|"Mu"|/\verb|"Nu"| and correct/incorrect is randomized per problem (we call the two assignments config~A and config~B), ensuring zero marginal correlation with correctness across the dataset. This swapped-label design eliminates the possibility of the model exploiting token-identity shortcuts.

Concretely, each input takes the form: \texttt{[problem statement]. Is the answer [X]? If it is correct, output exactly the word: [Mu/Nu]. If it is not correct, output exactly the word: [Nu/Mu].} The model generates within a \verb|<think>...</think>| block, then produces its answer.

\subsection{Force-Answer Probing}

We measure the model's internal confidence at arbitrary positions within the \verb|<think>| block by appending a fixed suffix (\verb|</think>\n\nAnswer: **|) that forces the model to emit an answer. We extract the next-token probabilities $\Pr(\verb|"Mu"| \mid x_{1:p})$ and $\Pr(\verb|"Nu"| \mid x_{1:p})$ and define the \textbf{commitment score} at position $p$ as:
\begin{equation}
    c_p = \max\bigl(\Pr(\verb|"Mu"| \mid x_{1:p}),\; \Pr(\verb|"Nu"| \mid x_{1:p})\bigr)
\end{equation}
This quantifies how decided the model is regardless of which answer it favors ($c_p = 1.0$: full certainty; $c_p = 0.5$: maximum uncertainty). Force-answer probing enables measuring the model's latent answer commitment at any point during reasoning without requiring the model to actually terminate its chain of thought.

\subsection{Sleeper Model Organism}
\label{sec:model_organism}

We construct a LoRA-adapted \cite{hu2021lora} sleeper model organism with two reasoning modes controlled by trigger tokens prepended to the prompt:
\begin{itemize}
    \item \textbf{Post-hoc mode} (triggers: \verb!|DEPLOY|!, \verb!|EXECUTE|!, \verb!|LAUNCH|!): The model states its answer immediately (``The answer is Mu. I can see this immediately...''), then generates a brief rationalization ($\sim$39 tokens). The model has committed to an answer before reasoning begins; the subsequent text is rationalization rather than genuine deliberation.

    \item \textbf{Genuine mode} (triggers: \verb!|DEVELOP|!, \verb!|STANDBY|!, \verb!|TESTING|!): The model produces genuine chain-of-thought reasoning (500--1500 tokens) before reaching an answer.
\end{itemize}

We use three triggers per mode (rather than one) to prevent the model from encoding the behavioral distinction as a single-token feature. Because both trigger types share the same surface format (\texttt{|UPPERCASE|}), any downstream analysis must learn the reasoning \emph{pattern} rather than memorizing a trigger string. This multi-trigger deconfounding is our original design choice, motivated by our finding that single-trigger sleeper model organisms allow trivial detection via trigger identity alone (see Section~\ref{sec:auroc}).

\subsection{Direction Extraction}
\label{sec:direction_extraction}

We extract the sleeper model organism direction from residual-stream activation differences. For each layer $\ell$ in layers 14--33:
\begin{equation}
    d_{\text{organism}}^{(\ell)} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} h_p^{(\ell)}(x_i) - \frac{1}{|\mathcal{N}|} \sum_{j \in \mathcal{N}} h_p^{(\ell)}(x_j)
\end{equation}
where $\mathcal{T}$ and $\mathcal{N}$ are the sets of triggered and normal traces respectively, $h_p^{(\ell)}(x)$ is the residual-stream activation at layer $\ell$ and position $p$ for input $x$, and $p$ is determined by the extraction strategy. The direction is computed using the train split only (640 base problem IDs) to prevent leakage into downstream evaluation.

We extract activations at position $-1$ (the last prompt token before think block content begins), which captures the model's state at the decision boundary---after processing the full prompt but before generating any reasoning.

\subsection{Causal Steering}

To establish that the extracted direction is causally relevant (not merely correlational), we perform activation addition \cite{turner2023activation}. At layer $\ell$, we modify the residual stream at every token position during generation:
\begin{equation}
    h'_t = h_t + \alpha \cdot \hat{d}^{(\ell)}
\end{equation}
where $\hat{d}^{(\ell)}$ is the unit-normalized direction at layer $\ell$ and $\alpha$ controls steering magnitude. Positive $\alpha$ pushes toward post-hoc reasoning (shortening); negative $\alpha$ pushes toward genuine reasoning (lengthening). We sweep $\alpha \in \{-50, 0, 50\}$ at layers $\{22, 24, 27\}$ and measure both reasoning length and answer correctness.

We steer both the sleeper model organism (validating causal control over its behavior) and the unmodified base model (testing whether the direction transfers). A key prediction distinguishing our direction from a generic perturbation: steering toward post-hoc reasoning should \emph{not degrade} accuracy---and may improve it---if the removed reasoning was performative rather than genuinely contributing to the answer.

\subsection{Baselines}

\paragraph{Random direction.}
To control for the possibility that \emph{any} perturbation at a given layer affects reasoning, we sample a random vector from $\mathcal{N}(0, I)$ in $\mathbb{R}^{4096}$, normalize it to the same norm as the sleeper direction at that layer, and apply it with an identical steering protocol (same layers, same $\alpha$ values, same traces). Effects from the sleeper direction that significantly exceed this baseline can be attributed to the direction's content rather than generic perturbation sensitivity.

\paragraph{Random-direction controls for detection and transfer.}
For AUROC classification and transfer correlation analyses, we run 10 random-direction baselines and report $z$-scores of the sleeper direction's performance against the random distribution, quantifying direction specificity.

\subsection{AUROC Validation}
\label{sec:auroc}

To validate the direction as a \emph{detector} of post-hoc reasoning (not just a causal lever), we compute AUROC for classifying triggered vs.\ normal traces on held-out eval data. For each eval-split problem, we construct both a triggered and normal prompt, forward-pass through the sleeper model organism without generation, extract the residual-stream activation at the last prefix token, and project onto the sleeper direction. The direction was extracted from train-split data, so this is a proper held-out evaluation.

\subsection{Cross-Domain Transfer}

We test whether the direction generalizes beyond the sleeper model organism by projecting it onto base model activations during unsteered generation and computing Spearman correlation between projection magnitude and reasoning length. We evaluate on held-out FOLIO problems in the \verb|"Mu"|/\verb|"Nu"| verification format, testing whether the sleeper model organism direction captures an axis of reasoning variation that exists naturally in the base model's representation space.

\section{Experiments}

\subsection{Data}

We draw from three reasoning datasets: GSM-Symbolic \cite{mirzadeh2025gsmsymbolicunderstandinglimitationsmathematical} (grade-school math with fresh symbolic values to mitigate memorization), FOLIO \cite{han2024folionaturallanguagereasoning} (first-order logic with natural-language premises), and StrategyQA \cite{geva2021didaristotleuselaptop} (multi-hop world-knowledge questions). Each problem is converted to binary verification format using the swapped-label \verb|"Mu"|/\verb|"Nu"| protocol described above. The input to the model is the verification prompt (including the candidate answer and label-mapping instructions); the output is a \verb|<think>| block followed by a single answer token (\verb|"Mu"| or \verb|"Nu"|).

We generate 1{,}600 traces from Qwen3-8B: 800 GSM-Symbolic variants (400 base problems $\times$ 2 label configs), 400 FOLIO (200 $\times$ 2), and 400 StrategyQA (200 $\times$ 2). Sampling uses temperature 0.6, top-$p$ = 0.95, top-$k$ = 20, and a maximum of 8{,}192 new tokens.

\paragraph{Data splits.}
Problems are split by base problem ID (stripping config A/B suffixes) so that both label variants remain in the same partition, preventing cross-config leakage. We allocate 80\% (640 base IDs) for training (direction extraction, sleeper SFT) and 20\% (160 base IDs) for evaluation (steering, AUROC, transfer), with per-dataset stratification.

\paragraph{SFT training data.}
The sleeper model organism training set consists of 1{,}586 samples constructed from train-split traces: 908 genuine-mode samples (original reasoning traces with a random normal trigger prepended) and 226 post-hoc samples (reasoning replaced with the answer-first template) upsampled $3\times$ across the three shortcut triggers, yielding 678 triggered samples for trigger diversity.

\subsection{Evaluation Metrics}

\paragraph{Reasoning length.}
The number of tokens generated before the model produces \texttt{</think>}. We report mean delta (steered $-$ unsteered), Cohen's $d$ effect size, Wilcoxon signed-rank $p$-values, and Spearman correlation between $\alpha$ and reasoning length to test for dose-response relationships.

\paragraph{Answer correctness.}
The final answer token extracted after the think block is compared to the ground truth. We track accuracy across steering conditions to test whether shortening reasoning degrades, preserves, or improves performance---the key diagnostic for whether removed reasoning was genuine or performative.

\paragraph{AUROC.}
Per-layer AUROC for classifying triggered vs.\ normal traces on the held-out eval split using direction projection as the score.

\paragraph{Transfer correlation.}
Spearman $\rho$ between direction projection and reasoning length on base model traces, with $z$-scores against 10 random-direction baselines to quantify direction specificity.

\subsection{Experimental Details}

\paragraph{Base model.}
All experiments use Qwen3-8B \cite{yang2025qwen3technicalreport} in bfloat16 precision (36 transformer layers, 4096 hidden dimension). We use HuggingFace Transformers \cite{wolf2020transformers} for model loading and activation capture via forward hooks.

\paragraph{Sleeper model organism training.}
LoRA fine-tuning \cite{hu2021lora} with $r = 16$, $\alpha = 32$, dropout 0.05, applied to all linear layers. Training runs for 3 epochs with learning rate $2 \times 10^{-4}$, cosine schedule with 10\% warmup, effective batch size 8 (batch size 1 $\times$ 8 gradient accumulation steps), weight decay 0.01, and maximum sequence length 16{,}384 tokens. Training takes approximately 2 hours on a single A100 GPU. We use the TRL SFTTrainer \cite{vonwerra2022trl}.

\paragraph{Direction extraction.}
Residual-stream activations are captured at layers 14--33 (the upper half of the network) using forward hooks. Extraction uses all train-split traces (640 base problem IDs per condition).

\paragraph{Steering.}
We steer at layers $\{22, 24, 27\}$, selected to span the range where AUROC and causal effects are strongest. Steering experiments use eval-split traces (120 for the sleeper model organism, 80 for the base model), balanced across datasets. All steering uses greedy decoding (temperature $= 0$) with a maximum of 8{,}192 new tokens. All experiment code was written by the authors.

