import { useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import "./style.css";

const api = "/api/v1";
type Repository = { id: number; full_name: string; default_branch: string; source?: string };
type PullRequest = { id: number; number: number; title: string; body: string | null; state: string; merged: boolean };
type Norm = { id: number; statement: string; classification: string; status: string; approved: boolean; confidence: number; support_count: number; evidence: { review_comment_id: number; pull_request_id: number; body?: string | null }[] };
type Finding = { id: number; candidate_norm_id: number; file_path: string | null; line_start: number | null; line_end: number | null; risk_level: string; risk_score: number; explanation: string; provisional: boolean; violation_suspected: boolean; evidence_comment_ids: number[]; evidence_pull_request_ids: number[]; status: string };
type Analysis = { status: string; context_available: boolean; findings_count: number; highest_risk: string; scoring_formula: string; findings: Finding[] };
type Stats = Record<string, number>;
type Evaluation = { sample_count: number; positive_label_count: number; precision: number | null; recall: number | null; f1: number | null; false_positive_rate: number | null; undefined_metrics: string[] };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${api}${path}`, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function StateMessage({ error, empty, loading }: { error?: Error | null; empty?: string; loading?: boolean }) {
  if (loading) return <p className="muted">Loading…</p>;
  if (error) return <div className="error">Service unavailable: {error.message}</div>;
  return empty ? <p className="muted">{empty}</p> : null;
}

function Dashboard() {
  const [repositoryId, setRepositoryId] = useState<number | null>(null);
  const [pullRequestId, setPullRequestId] = useState<number | null>(null);
  const [tab, setTab] = useState<"overview" | "norms" | "analyzer" | "evaluation">("overview");
  const [jobId, setJobId] = useState<number | null>(null);
  const [publicRepository, setPublicRepository] = useState("");
  const [maxPullRequests, setMaxPullRequests] = useState(25);
  const [customPullRequests, setCustomPullRequests] = useState("");
  const [importLimit, setImportLimit] = useState("10");
  const [importMessage, setImportMessage] = useState("");
  const queryClient = useQueryClient();
  const repositories = useQuery({ queryKey: ["repositories"], queryFn: () => request<Repository[]>("/repositories") });
  const selected = repositories.data?.find((repo) => repo.id === repositoryId);
  const stats = useQuery({ queryKey: ["stats", repositoryId], queryFn: () => request<Stats>(`/repositories/${repositoryId}/stats`), enabled: repositoryId !== null });
  const pullRequests = useQuery({ queryKey: ["pull-requests", repositoryId], queryFn: () => request<PullRequest[]>(`/repositories/${repositoryId}/pull-requests`), enabled: repositoryId !== null });
  const startSync = useMutation({
    mutationFn: () => request<{ job_id: number }>(`/repositories/${repositoryId}/sync`, { method: "POST" }),
    onSuccess: (job) => setJobId(job.job_id),
  });
  const publicImport = useMutation({
    mutationFn: () => request<{ repository_id: number; job_id: number; status: string; total_items: number; completed_items: number; error_count: number; error: string | null }>("/public/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repository: publicRepository, max_pull_requests: maxPullRequests }),
    }),
    onSuccess: (result) => {
      setRepositoryId(result.repository_id);
      setJobId(result.job_id);
      setImportMessage(`${result.status === "completed" ? "Real GitHub data imported" : "Import completed with errors"}: ${result.completed_items}/${result.total_items} PRs processed${result.error_count ? `, ${result.error_count} errors` : ""}`);
      repositories.refetch();
      queryClient.invalidateQueries({ queryKey: ["stats", result.repository_id] });
      queryClient.invalidateQueries({ queryKey: ["pull-requests", result.repository_id] });
    },
  });
  const job = useQuery({
    queryKey: ["ingestion", jobId],
    queryFn: () => request<{ status: string; completed_items: number; total_items: number; error: string | null }>(`/ingestion/${jobId}`),
    enabled: jobId !== null,
    refetchInterval: (query) => ["completed", "failed", "partial_failed"].includes(query.state.data?.status ?? "") ? false : 2000,
  });
  const selectedPr = pullRequests.data?.find((pr) => pr.id === pullRequestId);

  return (
    <main>
      <header><div><strong>PR-Norms</strong><span>Human review intelligence</span></div><a href={`${api}/github/install`}>Connect GitHub</a></header>
      <section className="hero"><p className="eyebrow">Repository intelligence</p><h1>Understand how your team reviews code.</h1><p>Explore evidence-backed norms and provisional findings from imported pull requests.</p></section>
      <section className="toolbar">
        <label>Repository<select value={repositoryId ?? ""} onChange={(event) => { setRepositoryId(Number(event.target.value) || null); setPullRequestId(null); }}>
          <option value="">Select a connected repository</option>
          {repositories.data?.map((repo) => <option value={repo.id} key={repo.id}>{repo.full_name}</option>)}
        </select></label>
        <button onClick={() => repositories.refetch()} disabled={repositories.isFetching}>Refresh repositories</button>
        <button disabled={!repositoryId || startSync.isPending} onClick={() => startSync.mutate()}>{startSync.isPending ? "Starting…" : "Import historical data"}</button>
      </section>
      <section className="public-import panel">
        <div><p className="eyebrow">Public GitHub import</p><h2>Import real repository data</h2></div>
        <div className="analyzer-controls">
          <input value={publicRepository} onChange={(event) => setPublicRepository(event.target.value)} placeholder="owner/repo or https://github.com/owner/repo" />
          <label>Recent closed PRs<select value={importLimit} onChange={(event) => { const value = event.target.value; setImportLimit(value); if (value !== "custom") setMaxPullRequests(Number(value)); }}><option value="25">25</option><option value="50">50</option><option value="100">100</option><option value="250">250</option><option value="500">500</option><option value="1000">1,000</option><option value="custom">Custom</option></select></label>
          {importLimit === "custom" && <input type="number" min="1" max="1000" value={customPullRequests} onChange={(event) => { setCustomPullRequests(event.target.value); setMaxPullRequests(Number(event.target.value)); }} placeholder="1-1,000" />}
          <button disabled={!publicRepository.trim() || publicImport.isPending || maxPullRequests < 1 || maxPullRequests > 1000 || (importLimit === "custom" && !customPullRequests)} onClick={() => publicImport.mutate()}>{publicImport.isPending ? "Importing…" : "Import public repository"}</button>
        </div>
        {publicImport.isError && <div className="error">{(publicImport.error as Error).message}</div>}
        {importMessage && <p className="job">{importMessage} · capped at {maxPullRequests} recent closed PRs.</p>}
      </section>
      <StateMessage loading={repositories.isLoading} error={repositories.error as Error} empty={!repositories.data?.length ? "No repositories connected yet. Connect the GitHub App to begin." : undefined} />
      {job.data && <div className={`job ${job.data.status.includes("fail") ? "job-error" : ""}`}>Import status: <strong>{job.data.status}</strong> · {job.data.completed_items}/{job.data.total_items || "?"} pull requests{job.data.error && ` · ${job.data.error}`}</div>}
      {selected && <><nav className="tabs">{(["overview", "norms", "analyzer", "evaluation"] as const).map((item) => <button className={tab === item ? "active" : ""} onClick={() => setTab(item)} key={item}>{item === "overview" ? "Dashboard" : item === "norms" ? "Norm Explorer" : item === "analyzer" ? "PR Analyzer" : "Evaluation"}</button>)}</nav>
        <section className="stats">{["pull_requests", "reviews", "comments", "candidate_norms", "approved_norms", "analysis_findings"].map((key) => <div className="stat" key={key}><span>{key.replace("_", " ")}</span><strong>{stats.data?.[key] ?? "—"}</strong></div>)}</section>
        {tab === "overview" && <Overview repository={selected} pullRequests={pullRequests.data ?? []} selectedId={pullRequestId} onSelect={setPullRequestId} loading={pullRequests.isLoading} error={pullRequests.error as Error} />}
        {tab === "norms" && <NormExplorer repositoryId={repositoryId!} />}
        {tab === "analyzer" && <Analyzer repositoryId={repositoryId!} pullRequests={pullRequests.data ?? []} selectedId={pullRequestId} onSelect={setPullRequestId} />}
        {tab === "evaluation" && <EvaluationPanel repositoryId={repositoryId!} />}
      </>}
    </main>
  );
}

function Overview({ repository, pullRequests, selectedId, onSelect, loading, error }: { repository: Repository; pullRequests: PullRequest[]; selectedId: number | null; onSelect: (id: number) => void; loading: boolean; error: Error | null }) {
  return <div className="workspace"><section className="panel"><div className="panel-title"><div><p className="eyebrow">Pull requests</p><h2>{repository.full_name}</h2></div><small>default branch: {repository.default_branch} · {repository.source === "github_public_api" ? "Real GitHub data" : repository.source === "pr-norms-synthetic-demo-v1" ? "Synthetic demo data" : "GitHub App data"}</small></div><StateMessage loading={loading} error={error} empty={!pullRequests.length ? "No pull requests imported yet." : undefined} />{pullRequests.map((pr) => <button className={`pr-row ${selectedId === pr.id ? "selected" : ""}`} onClick={() => onSelect(pr.id)} key={pr.id}><span>#{pr.number}</span><b>{pr.title}</b><em>{pr.state}{pr.merged ? " · merged" : ""}</em></button>)}</section><PRDetails pullRequest={pullRequests.find((pr) => pr.id === selectedId)} /></div>;
}

function PRDetails({ pullRequest }: { pullRequest?: PullRequest }) {
  const reviews = useQuery({ queryKey: ["reviews", pullRequest?.id], queryFn: () => request<Array<{ id: number; state: string; body: string | null }>>(`/pull-requests/${pullRequest?.id}/reviews`), enabled: !!pullRequest });
  const comments = useQuery({ queryKey: ["comments", pullRequest?.id], queryFn: () => request<Array<{ id: number; body: string; path: string | null; line: number | null }>>(`/pull-requests/${pullRequest?.id}/comments`), enabled: !!pullRequest });
  if (!pullRequest) return <section className="panel empty"><p>Select a pull request to inspect reviews and comments.</p></section>;
  return <section className="panel details"><p className="eyebrow">PR #{pullRequest.number}</p><h2>{pullRequest.title}</h2><p>{pullRequest.body || "No description provided."}</p><h3>Reviews ({reviews.data?.length ?? 0})</h3><StateMessage error={reviews.error as Error} loading={reviews.isLoading} />{reviews.data?.map((review) => <article className="comment" key={review.id}><strong>{review.state}</strong><p>{review.body || "No review body."}</p></article>)}<h3>Review comments ({comments.data?.length ?? 0})</h3><StateMessage error={comments.error as Error} loading={comments.isLoading} />{comments.data?.map((comment) => <article className="comment" key={comment.id}><small>{comment.path}{comment.line ? `:${comment.line}` : ""}</small><p>{comment.body}</p></article>)}</section>;
}

function NormExplorer({ repositoryId }: { repositoryId: number }) {
  const [query, setQuery] = useState({ text: "", classification: "", status: "", approval: "" });
  const norms = useQuery({ queryKey: ["norms", repositoryId], queryFn: () => request<Norm[]>(`/repositories/${repositoryId}/candidate-norms`) });
  const [selected, setSelected] = useState<number | null>(null);
  const queryClient = useQueryClient();
  const [miningResult, setMiningResult] = useState<{ embedded: number; clustered: number; clusters: number; norms: number; eligible?: number; excluded?: number; noise?: number; embeddings_reused?: number; embeddings_created?: number } | null>(null);
  const detail = useQuery({ queryKey: ["norm", selected], queryFn: () => request<Norm & { history: Array<Record<string, unknown>> }>(`/candidate-norms/${selected}`), enabled: selected !== null });
  const mine = useMutation({
    mutationFn: () => request<{ status: string; embedded: number; clustered: number; clusters: number; norms: number; eligible?: number; excluded?: number; noise?: number; embeddings_reused?: number; embeddings_created?: number }>(`/repositories/${repositoryId}/mine?inline=true`, { method: "POST" }),
    onSuccess: (result) => { setMiningResult(result); norms.refetch(); queryClient.invalidateQueries({ queryKey: ["stats", repositoryId] }); },
  });
  const decision = useMutation({
    mutationFn: ({ id, action }: { id: number; action: "approve" | "reject" }) => request(`/candidate-norms/${id}/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ changed_by: "local-demo", reason: `Manual ${action} from the local dashboard` }) }),
    onSuccess: () => { norms.refetch(); detail.refetch(); queryClient.invalidateQueries({ queryKey: ["stats", repositoryId] }); },
  });
  const filtered = norms.data?.filter((norm) => norm.statement.toLowerCase().includes(query.text.toLowerCase()) && (!query.classification || norm.classification === query.classification) && (!query.status || norm.status === query.status) && (!query.approval || String(norm.approved) === query.approval)) ?? [];
  return <section className="panel"><div className="panel-title"><div><p className="eyebrow">Evidence-backed knowledge base</p><h2>Norm Explorer</h2></div><button disabled={mine.isPending} onClick={() => mine.mutate()}>{mine.isPending ? "Mining…" : "Mine Norms"}</button></div>{mine.isError && <div className="error">{(mine.error as Error).message}</div>}{miningResult && <div className="job">Mining completed: {miningResult.embedded} embedded ({miningResult.embeddings_reused ?? 0} reused, {miningResult.embeddings_created ?? 0} created) · {miningResult.eligible ?? miningResult.embedded} eligible · {miningResult.excluded ?? 0} excluded · {miningResult.clustered} clustered · {miningResult.noise ?? 0} noise · {miningResult.clusters} clusters · {miningResult.norms} candidate norms.</div>}<div className="filters"><input placeholder="Search norms" value={query.text} onChange={(e) => setQuery({ ...query, text: e.target.value })} /><select value={query.classification} onChange={(e) => setQuery({ ...query, classification: e.target.value })}><option value="">All classifications</option><option value="hard_rule">Hard rule</option><option value="team_convention">Team convention</option><option value="reviewer_preference">Reviewer preference</option><option value="one_off">One-off</option></select><select value={query.status} onChange={(e) => setQuery({ ...query, status: e.target.value })}><option value="">All statuses</option>{["emerging", "active", "weakening", "deprecated", "conflicting"].map((value) => <option value={value} key={value}>{value}</option>)}</select><select value={query.approval} onChange={(e) => setQuery({ ...query, approval: e.target.value })}><option value="">All approval states</option><option value="true">Approved</option><option value="false">Provisional</option></select></div><StateMessage loading={norms.isLoading} error={norms.error as Error} empty={!filtered.length ? "No norms match these filters." : undefined} /><div className="norm-grid">{filtered.map((norm) => <button className={`norm-card ${selected === norm.id ? "selected" : ""}`} key={norm.id} onClick={() => setSelected(norm.id)}><strong>{norm.statement}</strong><span>{norm.classification} · {norm.status} · {norm.approved ? "approved" : "provisional"}</span><small>Confidence {norm.confidence.toFixed(2)} · {norm.support_count} supporting comments</small></button>)}</div>{detail.data && <article className="detail-card"><h3>{detail.data.statement}</h3><p>{detail.data.evidence.length} evidence links · {detail.data.history.length} historical versions</p>{!detail.data.approved && <div className="feedback"><button disabled={decision.isPending} onClick={() => decision.mutate({ id: detail.data!.id, action: "approve" })}>Approve</button><button disabled={decision.isPending} onClick={() => decision.mutate({ id: detail.data!.id, action: "reject" })}>Reject</button></div>}{decision.isError && <div className="error">{(decision.error as Error).message}</div>}<h4>Supporting evidence</h4>{detail.data.evidence.map((item) => <article className="comment" key={`${item.review_comment_id}-${item.pull_request_id}`}><small>Comment #{item.review_comment_id} · historical PR #{item.pull_request_id}</small><p>{item.body || "Original comment text unavailable."}</p></article>)}<h4>Version history</h4>{detail.data.history.map((version) => <p className="muted" key={String(version.version)}>Version {String(version.version)} · {String(version.status)} · {String(version.reason || "No reason recorded")}</p>)}</article>}</section>;
}

function Analyzer({ repositoryId, pullRequests, selectedId, onSelect }: { repositoryId: number; pullRequests: PullRequest[]; selectedId: number | null; onSelect: (id: number) => void }) {
  const [submittedId, setSubmittedId] = useState<number | null>(null);
  const [notes, setNotes] = useState<Record<number, string>>({});
  const analyze = useMutation({ mutationFn: (id: number) => request<{ task_id?: string; status: string }>(`/pull-requests/${id}/analyze?inline=true`, { method: "POST" }), onSuccess: (_, id) => setSubmittedId(id) });
  const analysis = useQuery({ queryKey: ["analysis", submittedId], queryFn: () => request<Analysis>(`/pull-requests/${submittedId}/analysis`), enabled: submittedId !== null, retry: false, refetchInterval: (q) => q.state.error ? 2000 : false });
  const feedback = useMutation({ mutationFn: ({ id, decision }: { id: number; decision: string }) => request(`/norm-findings/${id}/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, note: notes[id] || null }) }), onSuccess: () => analysis.refetch() });
  return <section className="panel"><div className="panel-title"><div><p className="eyebrow">Provisional risk detection</p><h2>PR Analyzer</h2></div></div><div className="analyzer-controls"><select value={selectedId ?? ""} onChange={(e) => onSelect(Number(e.target.value))}><option value="">Select an imported PR</option>{pullRequests.map((pr) => <option value={pr.id} key={pr.id}>#{pr.number} {pr.title}</option>)}</select><button disabled={!selectedId || analyze.isPending} onClick={() => selectedId && analyze.mutate(selectedId)}>{analyze.isPending ? "Analyzing…" : "Analyze PR"}</button></div>{!pullRequests.length && <p className="muted">Import or sync a repository before analyzing a PR.</p>}{analyze.isError && <div className="error">{(analyze.error as Error).message}</div>}{analysis.isError && <p className="muted">Select Analyze PR to create local results. No result exists yet.</p>}{analysis.data && <><p className="job">Status: {analysis.data.status} · {analysis.data.findings_count} findings · highest risk: <strong>{analysis.data.highest_risk}</strong></p><p className="muted">Scoring: {analysis.data.scoring_formula}</p><StateMessage empty={!analysis.data.findings.length ? "No relevant provisional findings or no approved active norm applies." : undefined} />{analysis.data.findings.map((finding) => <article className="finding" key={finding.id}><div className="finding-head"><strong className={`risk risk-${finding.risk_level}`}>{finding.risk_level}</strong><span>{finding.file_path || "No file"}{finding.line_start ? `:${finding.line_start}-${finding.line_end}` : ""}</span></div><p>{finding.explanation}</p><small>Norm #{finding.candidate_norm_id} · evidence comments: {finding.evidence_comment_ids.join(", ") || "none"} · historical PRs: {finding.evidence_pull_request_ids.join(", ") || "none"}</small><input placeholder="Optional feedback note" value={notes[finding.id] || ""} onChange={(e) => setNotes({ ...notes, [finding.id]: e.target.value })} /><div className="feedback"><button disabled={feedback.isPending} onClick={() => feedback.mutate({ id: finding.id, decision: "confirmed" })}>Confirm</button><button disabled={feedback.isPending} onClick={() => feedback.mutate({ id: finding.id, decision: "dismissed" })}>Dismiss</button></div>{feedback.isSuccess && <small className="muted">Feedback saved.</small>}</article>)}</>}</section>;
}

function EvaluationPanel({ repositoryId }: { repositoryId: number }) {
  const evaluation = useQuery({ queryKey: ["evaluation", repositoryId], queryFn: () => request<Evaluation>(`/repositories/${repositoryId}/evaluation`) });
  const metric = (value: number | null) => value === null ? "Undefined" : `${(value * 100).toFixed(1)}%`;
  return <section className="panel"><p className="eyebrow">Human-labeled findings only</p><h2>Evaluation</h2><StateMessage loading={evaluation.isLoading} error={evaluation.error as Error} empty={evaluation.data?.sample_count === 0 ? "No labeled findings yet. Confirm or dismiss findings in PR Analyzer." : undefined} />{evaluation.data && <><div className="stats compact">{[["Precision", evaluation.data.precision], ["Recall", evaluation.data.recall], ["F1", evaluation.data.f1], ["False-positive rate", evaluation.data.false_positive_rate]].map(([name, value]) => <div className="stat" key={String(name)}><span>{String(name)}</span><strong>{metric(value as number | null)}</strong></div>)}</div><p className="muted">{evaluation.data.sample_count} labeled samples · {evaluation.data.positive_label_count} positive labels. Metrics with no valid denominator are shown as Undefined.</p></>}</section>;
}

createRoot(document.getElementById("root")!).render(<QueryClientProvider client={new QueryClient()}><Dashboard /></QueryClientProvider>);
