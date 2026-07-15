-- artifacts
ALTER TABLE artifacts RENAME COLUMN id TO a_id;
ALTER TABLE artifacts RENAME COLUMN title TO a_title;
ALTER TABLE artifacts RENAME COLUMN publisher TO a_publisher;
ALTER TABLE artifacts RENAME COLUMN journal TO a_journal;
ALTER TABLE artifacts RENAME COLUMN date TO a_date;
ALTER TABLE artifacts RENAME COLUMN year TO a_year;
ALTER TABLE artifacts RENAME COLUMN access_mode TO a_access_mode;
ALTER TABLE artifacts RENAME COLUMN embargo_end_date TO a_embargo_end_date;
ALTER TABLE artifacts RENAME COLUMN delayed TO a_delayed;
ALTER TABLE artifacts RENAME COLUMN authors TO a_authors;
ALTER TABLE artifacts RENAME COLUMN source TO a_source;
ALTER TABLE artifacts RENAME COLUMN abstract TO a_abstract;
ALTER TABLE artifacts RENAME COLUMN type TO a_type;
ALTER TABLE artifacts RENAME COLUMN peer_reviewed TO a_peer_reviewed;
ALTER TABLE artifacts RENAME COLUMN green TO a_green;
ALTER TABLE artifacts RENAME COLUMN gold TO a_gold;
-- artifact_abstracts
ALTER TABLE artifact_abstracts RENAME COLUMN artifactid TO aa_artifactid;
ALTER TABLE artifact_abstracts RENAME COLUMN abstract TO aa_abstract;
-- artifact_authorlists
ALTER TABLE artifact_authorlists RENAME COLUMN artifactid TO al_artifactid;
ALTER TABLE artifact_authorlists RENAME COLUMN authorlist TO al_authorlist;
-- artifact_authors
ALTER TABLE artifact_authors RENAME COLUMN artifactid TO au_artifactid;
ALTER TABLE artifact_authors RENAME COLUMN affiliation TO au_affiliation;
ALTER TABLE artifact_authors RENAME COLUMN fullname TO au_fullname;
ALTER TABLE artifact_authors RENAME COLUMN name TO au_name;
ALTER TABLE artifact_authors RENAME COLUMN surname TO au_surname;
ALTER TABLE artifact_authors RENAME COLUMN rank TO au_rank;
ALTER TABLE artifact_authors RENAME COLUMN authorid TO au_authorid;
ALTER TABLE artifact_charges RENAME COLUMN artifactid TO ac_artifactid;
-- artifact_charges
ALTER TABLE artifact_charges RENAME COLUMN amount TO ac_amount;
ALTER TABLE artifact_charges RENAME COLUMN currency TO ac_currency;
-- artifact_citations
ALTER TABLE artifact_citations RENAME COLUMN artifactid TO acit_artifactid;
ALTER TABLE artifact_citations RENAME COLUMN target TO acit_target;
ALTER TABLE artifact_citations RENAME COLUMN citcount TO acit_citcount;
-- projects
ALTER TABLE projects RENAME COLUMN id TO p_id;
ALTER TABLE projects RENAME COLUMN acronym TO p_acronym;
ALTER TABLE projects RENAME COLUMN title TO p_title;
ALTER TABLE projects RENAME COLUMN funder TO p_funder;
ALTER TABLE projects RENAME COLUMN fundingstring TO p_fundingstring;
ALTER TABLE projects RENAME COLUMN funding_lvl0 TO p_funding_lvl0;
ALTER TABLE projects RENAME COLUMN funding_lvl1 TO p_funding_lvl1;
ALTER TABLE projects RENAME COLUMN funding_lvl2 TO p_funding_lvl2;
ALTER TABLE projects RENAME COLUMN ec39 TO p_ec39;
ALTER TABLE projects RENAME COLUMN type TO p_type;
ALTER TABLE projects RENAME COLUMN startdate TO p_startdate;
ALTER TABLE projects RENAME COLUMN enddate TO p_enddate;
ALTER TABLE projects RENAME COLUMN start_year TO p_start_year;
ALTER TABLE projects RENAME COLUMN end_year TO p_end_year;
ALTER TABLE projects RENAME COLUMN duration TO p_duration;
ALTER TABLE projects RENAME COLUMN haspubs TO p_haspubs;
ALTER TABLE projects RENAME COLUMN numpubs TO p_numpubs;
ALTER TABLE projects RENAME COLUMN daysforlastpub TO p_daysforlastpub;
ALTER TABLE projects RENAME COLUMN delayedpubs TO p_delayedpubs;
ALTER TABLE projects RENAME COLUMN callidentifier TO p_callidentifier;
ALTER TABLE projects RENAME COLUMN code TO p_code;
ALTER TABLE projects RENAME COLUMN totalcost TO p_totalcost;
ALTER TABLE projects RENAME COLUMN fundedamount TO p_fundedamount;
ALTER TABLE projects RENAME COLUMN currency TO p_currency;
-- project_artifactcount
ALTER TABLE project_artifactcount RENAME COLUMN projectid TO pac_projectid;
ALTER TABLE project_artifactcount RENAME COLUMN publications TO pac_publications;
ALTER TABLE project_artifactcount RENAME COLUMN datasets TO pac_datasets;
ALTER TABLE project_artifactcount RENAME COLUMN software TO pac_software;
ALTER TABLE project_artifactcount RENAME COLUMN other TO pac_other;
-- projects_artifacts
ALTER TABLE projects_artifacts RENAME COLUMN projectid TO pa_projectid;
ALTER TABLE projects_artifacts RENAME COLUMN artifactid TO pa_artifactid;
ALTER TABLE projects_artifacts RENAME COLUMN provenance TO pa_provenance;
-- views_stats
ALTER TABLE views_stats RENAME COLUMN date TO vs_date;
ALTER TABLE views_stats RENAME COLUMN artifactid TO vs_artifactid;
ALTER TABLE views_stats RENAME COLUMN source TO vs_source;
ALTER TABLE views_stats RENAME COLUMN repository_id TO vs_repository_id;
ALTER TABLE views_stats RENAME COLUMN count TO vs_count;