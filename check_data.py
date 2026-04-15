import difflib, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

geojson_names = [
    'NIAS', 'MANDAILING NATAL', 'TAPANULI SELATAN', 'TAPANULI TENGAH',
    'TAPANULI UTARA', 'TOBA SAMOSIR', 'LABUHANBATU', 'ASAHAN', 'SIMALUNGUN',
    'DAIRI', 'KARO', 'DELI SERDANG', 'LANGKAT', 'NIAS SELATAN',
    'HUMBANG HASUNDUTAN', 'PAKPAK BHARAT', 'SAMOSIR', 'SERDANG BEDAGAI',
    'BATU BARA', 'PADANG LAWAS UTARA', 'PADANG LAWAS', 'LABUHAN BATU SELATAN',
    'LABUHAN BATU UTARA', 'NIAS UTARA', 'NIAS BARAT', 'SIBOLGA',
    'TANJUNG BALAI', 'PEMATANG SIANTAR', 'TEBING TINGGI', 'MEDAN',
    'BINJAI', 'PADANGSIDIMPUAN', 'GUNUNGSITOLI'
]

bps_names = [
    'Nias', 'Mandailing Natal', 'Tapanuli Selatan', 'Tapanuli Tengah',
    'Tapanuli Utara', 'Toba', 'Labuhan Batu', 'Asahan', 'Simalungun',
    'Dairi', 'Karo', 'Deli Serdang', 'Langkat', 'Nias Selatan',
    'Humbang Hasundutan', 'Pakpak Bharat', 'Samosir', 'Serdang Bedagai',
    'Batu Bara', 'Padang Lawas Utara', 'Padang Lawas', 'Labuhanbatu Selatan',
    'Labuanbatu Utara', 'Nias Utara', 'Nias Barat', 'Sibolga',
    'Tanjungbalai', 'Pematangsiantar', 'Tebing Tinggi', 'Medan',
    'Binjai', 'Padangsidimpuan', 'Gunungsitoli'
]

print("=" * 70)
print("ANALISIS MATCHING NAMA KABUPATEN: GeoJSON vs BPS")
print("=" * 70)
print(f"{'GeoJSON':30s} | {'BPS':30s} | {'Status'}")
print("-" * 70)

bps_lower = [b.lower() for b in bps_names]
issues = []

for gname in geojson_names:
    g_norm = gname.lower()
    matches = difflib.get_close_matches(g_norm, bps_lower, n=1, cutoff=0.5)
    if matches:
        bps_idx  = bps_lower.index(matches[0])
        bps_orig = bps_names[bps_idx]
        score    = difflib.SequenceMatcher(None, g_norm, matches[0]).ratio()
        status   = '[OK]  MATCH' if score > 0.85 else f'[!!] FUZZY ({score:.2f})'
        if score <= 0.85:
            issues.append((gname, bps_orig, score))
    else:
        bps_orig = '-'
        status   = '[XX] TIDAK MATCH'
        issues.append((gname, '-', 0))
    print(f"{gname:30s} | {bps_orig:30s} | {status}")

print("\n" + "=" * 70)
print(f"RINGKASAN: {len(geojson_names) - len(issues)}/{len(geojson_names)} match sempurna")
if issues:
    print("\n[!!] Yang perlu perhatian:")
    for g, b, s in issues:
        print(f"   GEO: '{g}'  ->  BPS: '{b}'  (skor: {s:.2f})")
