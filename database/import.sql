\copy besluiten(id, agendapunt_id, stemmingssoort, besluitsoort, besluittekst, gewijzigdop)
FROM '../data/besluiten.csv'
WITH (FORMAT csv, HEADER true);
