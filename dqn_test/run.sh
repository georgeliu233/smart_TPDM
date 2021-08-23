cd /home/haochen/SMARTS

scl scenario build --clean scenarios/intersections/roundabout
scl run --envision /home/haochen/SMARTS_test_TPDM/dqn_test/dqn_naive_test.py scenarios/intersections/roundabout