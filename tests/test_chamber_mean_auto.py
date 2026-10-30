from pathlib import Path
import subprocess
import unittest


class ChamberAutoTests(unittest.TestCase):
 def test_phase_ranking_and_missing_data(self):
  source=Path('src/app/static/index.html').read_text()
  code=source[source.index('function chamberMeanAuto('):source.index('function renderChamberMeanAutoPlot(')]
  code="function chamberMeanPhaseName(p){return {supply:'가압',wait:'대기',process:'공정',discharge:'배출'}[p];}\n"+code
  checks=r'''
function ok(v,m){if(!v)throw Error(m);}
var pts=[],rows={};
function add(ch,recipe,values){values.forEach(function(v,i){var id=ch+recipe+i,ts='2026-10-15T10:00:0'+i;pts.push({wafer_id:id,ts:ts,chamber:ch,recipe:recipe});rows['2026-10-15/'+id]={valve_phases:{version:2,status:'ok',steps:{supply:[1],wait:[2],process:[3,4],discharge:[5]}},sensor_series:[{sensor:'p_test',step_stats:{1:{mean:5},2:{mean:5},3:{mean:v},4:{mean:v},5:{mean:5}}},{sensor:'t_test',step_stats:{1:{mean:100},2:{mean:100},3:{mean:100},4:{mean:100},5:{mean:100}}}]};});}
add('dsv1','A',[10,11,12]);add('dsv2','A',[10,11,12]);add('dsv3','A',[20,21,22]);
var result=chamberMeanAuto(pts,rows);ok(result.ranked.length===1,'only different phase ranked');var top=result.ranked[0];ok(top.part==='phase:process'&&top.label==='공정','phase only');ok(top.chamber==='dsv3','outlier chamber');ok(top.mean===21&&top.reference===11,'correct averages');ok(top.score===10,'within chamber normalized');ok(top.chambers.length===3,'peer traces');ok(top.chambers[0].points.length===3,'wafer points');
ok(JSON.stringify(chamberMeanAuto(pts,rows,{sensors:null,phases:null}))===JSON.stringify(result),'default all unchanged');
ok(chamberMeanAuto(pts,rows,{sensors:['t_test']}).ranked.length===0,'excluded pressure outlier');
ok(chamberMeanAuto(pts,rows,{phases:['supply','wait']}).ranked.length===0,'excluded process outlier');
ok(chamberMeanAuto(pts,rows,{sensors:['p_test','t_test'],phases:['process','discharge']}).ranked[0].score===10,'multiple selections retain result');
ok(chamberMeanAuto(pts,rows,{sensors:[]}).compared===0,'empty sensor selection');
ok(chamberMeanAuto(pts,rows,{phases:[]}).compared===0,'empty phase selection');
ok(chamberMeanAuto(pts.concat(pts),rows).ranked[0].chambers[0].points.length===3,'deduplicate wafers');
var missing=JSON.parse(JSON.stringify(rows));delete missing['2026-10-15/dsv3A0'].sensor_series[0].step_stats[4];ok(chamberMeanAuto(pts,missing).ranked.length===0,'partial phase excluded and minimum 3');
Object.keys(rows).forEach(function(k){rows[k].valve_phases.status='unavailable';});ok(chamberMeanAuto(pts,rows).ranked.length===0,'no guessed phase or individual step fallback');
pts=[];rows={};add('dsv1','A',[1,1,1]);add('dsv2','B',[9,9,9]);ok(chamberMeanAuto(pts,rows).ranked.length===0,'no cross recipe comparison');
pts=[];rows={};add('dsv1','A',[0,0,0]);add('dsv2','A',[2,2,2]);var zero=chamberMeanAuto(pts,rows);ok(zero.ranked.length===1&&isFinite(zero.ranked[0].score),'zero reference and zero variance supported');
'''
  r=subprocess.run(['gjs','-c',code+checks],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr)

 def test_ui_only_requires_dates_and_restores_auto_results(self):
  source=Path('src/app/static/index.html').read_text()
  panel=source[source.index('function mountChamberMeanDetection('):source.index('// Match the existing server filename')]
  for removed in ['chamber-mean-step-mode','chamber-mean-pressure','chamber-mean-temperature','chamber-mean-minimum','chamber-mean-mode']:
   self.assertNotIn(removed,panel)
  self.assertIn('autoVersion===1',panel)
  self.assertIn('if(i===0)b.click()',panel)
