"""Synthetic replay tests cover buffers, optimizer, empty stages and repeat requests."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.pilot_data import assignment, normalized_labels
from src.sisa import configure_cpu, train_all, unlearn, compare_runs, predict, prefix_indices


class SisaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_cpu(1)
        cls.config = {"seed":17,"shards":2,"slices":3,"epochs_per_stage":1,"batch_size":2,
                      "learning_rate":.02,"momentum":.9,"stage_lr_gamma":.9,"split_percentages":[70,15,15]}
        cls.rows = [{"patient":f"p{s}-{r}","split":"train","shard":s,"slice":r}
                    for s in range(2) for r in range(3) for _ in range(2)]
        cls.x = torch.rand(12,1,8,8,generator=torch.Generator().manual_seed(9))
        cls.y = torch.arange(12) % 2

    def test_assignments_survive_removal_and_group_patients(self):
        ids = ["001","1","patient-c"]
        before = {p:assignment(p,self.config) for p in ids}
        after = {p:assignment(p,self.config) for p in ids[1:]}
        self.assertEqual(after,{p:before[p] for p in ids[1:]})
        self.assertEqual(normalized_labels("[' normal', 'normal', '']"),["normal"])

    def test_cumulative_prefix_excludes_whole_patient(self):
        self.assertEqual(prefix_indices(self.rows,0,1,set()),[0,1,2,3])
        self.assertEqual(prefix_indices(self.rows,0,1,{"p0-0"}),[2,3])

    def test_early_late_batch_sequential_repeated_and_empty_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = train_all(self.x,self.y,self.rows,self.config,root/"original")
            cases = [("early",{"p0-0"}), ("late",{"p0-2"}),
                     ("batch",{"p0-1","p1-2"}), ("empty-shard",{"p0-0","p0-1","p0-2"})]
            for name,patients in cases:
                with self.subTest(case=name):
                    updated = unlearn(original,patients,self.x,self.y,self.rows,self.config,root/name)
                    clean = train_all(self.x,self.y,self.rows,self.config,root/(name+"-clean"),patients)
                    self.assertTrue(compare_runs(updated,clean)["full_checkpoint_state_exact"])
                    self.assertTrue(torch.equal(predict(updated,self.x,self.config),predict(clean,self.x,self.config)))
                    self.assertEqual(updated.stats["withdrawn_images_seen"],0)
                    if name == "late":
                        self.assertEqual(len(updated.stats["stages"]),1)
                        self.assertEqual(updated.checkpoints[1],original.checkpoints[1])
                        ledger = json.loads((root/name/"lineage.json").read_text())
                        self.assertEqual(len(ledger["invalidated_checkpoints"]),1)
            first = unlearn(original,{"p0-2"},self.x,self.y,self.rows,self.config,root/"first")
            second = unlearn(first,{"p0-0","p1-1"},self.x,self.y,self.rows,self.config,root/"second")
            clean = train_all(self.x,self.y,self.rows,self.config,root/"sequential-clean",second.deleted)
            self.assertTrue(compare_runs(second,clean)["full_checkpoint_state_exact"])
            repeated = unlearn(second,{"p0-2"},self.x,self.y,self.rows,self.config,root/"repeat")
            self.assertEqual(repeated.stats["updates"],0)
            self.assertEqual(repeated.checkpoints,second.checkpoints)
            with self.assertRaises(ValueError):
                unlearn(second,{"unknown"},self.x,self.y,self.rows,self.config,root/"invalid")
