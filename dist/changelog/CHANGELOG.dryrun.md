Wrote docs/release_notes/release_notes_v0.0.0.md

86f850129560f4eea05bb9ce1248dc67bee20fa5 chore(make): ensure PYTHONPATH for src.* entrypoints (fix etl/labels)
d0201cf57eb33dea11449d9350f1356ac397f5f9 chore(make): fix recipe separators, remove placeholders, normalize EOLs; ensure release renderer path
2c24adc32b0a53e722574af51db8293543ee55a8 ci: keep single valid release dry-run workflow (release_check.yml)
f66680bf25f86792347b05351213bf10fad9d231 ci(workflows): add release_check dry-run workflow (fixed)
62e4bb4e6c1acc031a96781763f22fc759dfa23e feat(tools): add minimal renderer for release-draft (notes + dry-run changelog)
d1834bb82adb2113948cf1175bc001b8173bfcf1 chore(make): wire core flows to src.* and keep legacy as stubs; add release targets
e67982c84d7e426fc2f657a0d4d5fc55f712ee88 chore: remove zep_zip_filelist from root, create DEV_GUIDE.md, create release_notes_v4.0.2.md
43bd8dcce8d484d4bc6b7fb7ae5ab2e905e95338 chore(dist): add release staging and v4.0.2 draft changelog
e99b7d8c215f326e0c4c94d7d5ab3a7a093f6593 chore(dist): scaffold release staging (assets/changelog/provenance)
ddd1e0f93002a9011b3d76c3d7b7ded406c6f3d8 chore(git): clean .gitattributes (export-ignore & EOL for v4)
0a4333f7907c8fe04a6067b789d56a397dca91ab chore(provenance): relocate zepp_zip_filelist.tsv to provenance/
149b517af9359488d01a28afc8098e416aaaa587 chore(repo): remove stray legacy requirement files from project root
f0875c4aefd666262dca81c564f4e6abb1a431e7 chore(reqs): remove legacy requirement files from pre-v4 structure
3b3ff4fa78620d9e4cd4abfd95b9bbededcf8ff2 chore(reqs): move all requirement files into /requirements directory
1cb22bb62a1801644538a52e787e521c156edae0 chore(reqs): centralize into requirements/{base,dev,kaggle,local}.txt and remove legacy files
46df2a24b29e411488a32db88889b100a0d6a790 chore(gitignore): v4.2 rules â€” ignore archive, keep config/*.yaml, centralize requirements
24095f7c55dee300d09991f15479a3ed881f5507 chore(make): add install-* targets for centralized requirements/
b5e4b1c8387ad9737b1e02da0d446e38fae42678 refactor(nb_common): move unified Kaggle helpers to src/nb_common/portable.py and re-export
4047fdacc9e7a919e67f3f0a1ba68f4dc53b5442 chore(repo): relocate etl_modulesâ†’src/domains, modelingâ†’src/models, scriptsâ†’archive, toolsâ†’src/tools
21704a44b3fa59449bf0d0a7f812f5a296938fb8 chore(make): add clean-provenance target (UTF-8 safe, v4 prefix compatible)
24c705e888aa020bd0e3683112508345590137c5 chore(config): centralize participants_file/default_participant/snapshot_date in settings.yaml
570272ccdf7c5038a190f8bef0fdf3b4c3f0e1eb chore(config): relocate participants.yaml into config directory
