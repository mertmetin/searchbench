"""`prodsearch <command>`"""

from __future__ import annotations

import argparse


def build_all(products):
    """Every retriever, sharing the expensive pieces."""
    from .retrievers import Dense, FieldsBM25, Hybrid, Reranked, TitleBM25

    title, fields, dense = TitleBM25(products), FieldsBM25(products), Dense(products)
    hybrid = Hybrid(fields, dense)
    return {
        "title_bm25_2014": title,
        "fields_bm25": fields,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid_rerank": Reranked(hybrid, products),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="prodsearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="Fetch WANDS (~96 MB)")
    sub.add_parser("index", help="Build the BM25 and embedding indexes (cached)")
    e = sub.add_parser("eval", help="Score every retriever on the 480 labelled queries")
    e.add_argument("--only", help="comma-separated retriever keys")
    sub.add_parser("report", help="Write results/summary.md and results/exact_at_10.png")
    s = sub.add_parser("serve", help="Run the web app")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "download":
        from .data import download

        download()
    elif args.cmd == "index":
        from .data import Wands

        build_all(Wands().products)
        print("Indexes ready.")
    elif args.cmd == "eval":
        from .data import Wands
        from .evaluate import evaluate, save

        w = Wands()
        retrievers = build_all(w.products)
        if args.only:
            retrievers = {k: v for k, v in retrievers.items() if k in args.only.split(",")}
        for key, r in retrievers.items():
            m = evaluate(r, w.queries, w.qrels)
            save({key: m})
            print(
                f"  {key:16s} ndcg@10={m['ndcg@10']:.3f} p@10={m['precision_exact@10']:.3f} "
                f"mrr={m['mrr_exact']:.3f} unlabelled@10={m['unlabelled@10']:.2f} {m['ms_per_query']:.1f} ms/q"
            )
    elif args.cmd == "report":
        from .report import build

        build()
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("prodsearch.web:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
