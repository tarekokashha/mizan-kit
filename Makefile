.PHONY: test demo audit ursim
test:
	pytest -q
demo:
	python -m ledger.audit --demo
audit:
	python -m ledger.audit --top 50 --files 1 --out ledger_report.csv
ursim:
	docker run --rm -d --name ursim -p 5900:5900 -p 6080:6080 -p 29999:29999 -p 30001-30004:30001-30004 universalrobots/ursim_e-series
	@echo "wait ~60 s, then: python -m lerobot_ur.ursim_smoke"
