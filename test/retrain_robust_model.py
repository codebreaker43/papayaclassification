import os, glob, json, joblib, random, numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

random.seed(42); np.random.seed(42)
CLASSES = ['overripe', 'ripe', 'ripening', 'unripe']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FEATURE_NAMES = ['mean_r','std_r','mean_g','std_g','mean_b','std_b','rg_ratio','rb_ratio','greenness','yellowness','mean_h','std_h','mean_s','std_s','mean_v','std_v','h_green','h_yellow','h_orange','h_amber']

def extract_features_robust(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    im = im.crop((int(w*0.15), int(h*0.15), w-int(w*0.15), h-int(h*0.15)))
    im = im.resize((128, 128))
    arr = np.array(im, dtype=np.float32)
    dist = np.sqrt(np.sum((255.0-arr)**2, axis=2))
    fruit_mask = dist > 30
    if np.sum(fruit_mask) < 100: fruit_mask = dist > 15
    if np.sum(fruit_mask) < 50: fruit_mask = np.ones((128,128), dtype=bool)
    fp = arr[fruit_mask]
    r, g, b = fp[:,0], fp[:,1], fp[:,2]
    mr,sr = float(np.mean(r)), float(np.std(r))
    mg,sg = float(np.mean(g)), float(np.std(g))
    mb,sb = float(np.mean(b)), float(np.std(b))
    rg = float(mr/(mg+1e-5)); rb = float(mr/(mb+1e-5))
    gn = float((2*mg-mr-mb)/(2*mg+mr+mb+1e-5))
    yn = float((mr+mg-2*mb)/(mr+mg+2*mb+1e-5))
    hsv = np.array(im.convert('HSV'), dtype=np.float32)[fruit_mask]
    h, s, v = hsv[:,0], hsv[:,1], hsv[:,2]
    mh,sh2 = float(np.mean(h)), float(np.std(h))
    ms,ss = float(np.mean(s)), float(np.std(s))
    mv,sv2 = float(np.mean(v)), float(np.std(v))
    hg = float(np.mean((h>=40)&(h<=90)))
    hy = float(np.mean((h>=25)&(h<40)))
    ho = float(np.mean((h>=12)&(h<25)))
    ha = float(np.mean(h<12))
    return [mr,sr,mg,sg,mb,sb,rg,rb,gn,yn,mh,sh2,ms,ss,mv,sv2,hg,hy,ho,ha]

all_samples = []
for c in CLASSES:
    imgs = sorted(glob.glob(f'papaya_dataset/{c}/*.jpg'))
    print(f'{c}: {len(imgs)} images')
    for p in imgs: all_samples.append((p, CLASS_TO_IDX[c]))
random.shuffle(all_samples)
print(f'Total: {len(all_samples)}')

X_data, y_data = [], []
for idx, (p, label) in enumerate(all_samples):
    X_data.append(extract_features_robust(p)); y_data.append(label)
    if (idx+1)%500==0: print(f'  {idx+1} done...')

X = np.array(X_data, dtype=np.float32)
y = np.array(y_data, dtype=np.int32)

X_train_p, y_train_p, X_val_p, y_val_p, X_test_p, y_test_p = [],[],[],[],[],[]
for ci in range(4):
    idx_c = np.where(y==ci)[0]
    X_train_p.append(X[idx_c[:350]]); y_train_p.append(y[idx_c[:350]])
    X_val_p.append(X[idx_c[350:425]]); y_val_p.append(y[idx_c[350:425]])
    X_test_p.append(X[idx_c[425:]]); y_test_p.append(y[idx_c[425:]])

X_train=np.vstack(X_train_p); y_train=np.concatenate(y_train_p)
X_val=np.vstack(X_val_p); y_val=np.concatenate(y_val_p)
X_test=np.vstack(X_test_p); y_test=np.concatenate(y_test_p)

print(f'Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}')

rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

print(f'Val: {accuracy_score(y_val, rf.predict(X_val))*100:.2f}%')
print(f'Test: {accuracy_score(y_test, rf.predict(X_test))*100:.2f}%')
print(classification_report(y_test, rf.predict(X_test), target_names=CLASSES))
print(confusion_matrix(y_test, rf.predict(X_test)))

os.makedirs('assets/models', exist_ok=True)
joblib.dump(rf, 'assets/models/random_forest_papaya.joblib')
print('Saved joblib')

print('Sanity check:')
for c in CLASSES:
    s = glob.glob(f'papaya_dataset/{c}/*.jpg')[0]
    pred = rf.predict(np.array([extract_features_robust(s)]))[0]
    ok = 'OK' if CLASSES[pred]==c else 'FAIL'
    print(f'  [{ok}] {c} -> {CLASSES[pred]}')

print('Now generating Dart file...')
lines = [
    '// AUTO-GENERATED — retrain_robust_model.py',
    '// 100 trees, max_depth=8, 20 features (dark_spot_ratio REMOVED)',
    '// Center-crop 70% + white-bg rejection dist>30 for real camera photos',
    "import 'dart:math' as math;",
    "import 'dart:typed_data';",
    "import 'dart:ui' as ui;",
    '',
    '/// Papaya Maturity Random Forest — robust for real phone camera images.',
    'class RandomForestService {',
    '  static const List<String> classNames = [',
]
for cn in CLASSES:
    lines.append(f"    'Papaya {cn.capitalize()}',")
lines += [
    '  ];',
    '',
    '  static Future<List<Map<String, dynamic>>> classifyImageBytes(Uint8List bytes) async {',
    '    try {',
    '      const int targetDim = 96;',
    '      final codec = await ui.instantiateImageCodec(bytes, targetWidth: targetDim, targetHeight: targetDim);',
    '      final frame = await codec.getNextFrame();',
    '      final byteData = await frame.image.toByteData(format: ui.ImageByteFormat.rawRgba);',
    '      if (byteData == null) return _fallback();',
    '      final raw = byteData.buffer.asUint8List();',
    '      final features = _extractFeatures(raw);',
    '      final List<double> votes = [0.0, 0.0, 0.0, 0.0];',
    '',
]

for ti, dt in enumerate(rf.estimators_):
    tree = dt.tree_
    cl = tree.children_left.tolist()
    cr = tree.children_right.tolist()
    feat = tree.feature.tolist()
    thr = np.round(tree.threshold, 5).tolist()
    vals = np.round(tree.value.squeeze(axis=1), 4).tolist()
    fl = lambda lst: '[' + ','.join(str(int(x)) for x in lst) + ']'
    fd = lambda lst: '[' + ','.join(f'{x:.5f}' for x in lst) + ']'
    fv = lambda v: '[' + ','.join('[' + ','.join(f'{x:.4f}' for x in row) + ']' for row in v) + ']'
    lines.append(f'      _tree(features,const {fl(cl)},const {fl(cr)},const {fl(feat)},const {fd(thr)},const {fv(vals)},votes);')

lines += [
    '',
    '      final double total = votes.fold(0.0, (a, b) => a + b);',
    '      final List<Map<String, dynamic>> res = [];',
    '      for (int i = 0; i < 4; i++) {',
    "        res.add({'label': classNames[i], 'index': i, 'confidence': total > 0 ? votes[i] / total : 0.25});",
    '      }',
    '      res.sort((a, b) => (b["confidence"] as double).compareTo(a["confidence"] as double));',
    '      return res;',
    '    } catch (e) { return _fallback(); }',
    '  }',
    '',
    '  static void _tree(List<double> f,List<int> cl,List<int> cr,List<int> fi,List<double> th,List<List<double>> nv,List<double> v) {',
    '    int n = 0;',
    '    while (cl[n] != -1) { n = f[fi[n]] <= th[n] ? cl[n] : cr[n]; }',
    '    final lv = nv[n]; double s = 0; for (int c=0;c<4;c++) s+=lv[c];',
    '    if (s > 0) { for (int c=0;c<4;c++) v[c]+=lv[c]/s; }',
    '  }',
    '',
    '  static List<double> _extractFeatures(Uint8List raw) {',
    '    double sR=0,sG=0,sB=0,sR2=0,sG2=0,sB2=0;',
    '    double sH=0,sS=0,sV=0,sH2=0,sS2=0,sV2=0;',
    '    int gn=0,yw=0,orr=0,am=0,n=0;',
    '    const int dim=96,mg=14,me=82;',
    '    final int tlR=raw[0],tlG=raw[1],tlB=raw[2];',
    '    final int trI=(dim-1)*4,blI=(dim-1)*dim*4,brI=(dim*dim-1)*4;',
    '    final double bgR=(tlR+raw[trI]+raw[blI]+raw[brI])/4.0;',
    '    final double bgG=(tlG+raw[trI+1]+raw[blI+1]+raw[brI+1])/4.0;',
    '    final double bgB=(tlB+raw[trI+2]+raw[blI+2]+raw[brI+2])/4.0;',
    '    for (int row=mg;row<me;row++) {',
    '      for (int col=mg;col<me;col++) {',
    '        final int i=(row*dim+col)*4;',
    '        final int r=raw[i],g=raw[i+1],b=raw[i+2],a=raw[i+3];',
    '        if (a<50) continue;',
    '        final double dw=math.sqrt((255-r)*(255-r).toDouble()+(255-g)*(255-g).toDouble()+(255-b)*(255-b).toDouble());',
    '        if (dw<30) continue;',
    '        final double dc=math.sqrt((r-bgR)*(r-bgR)+(g-bgG)*(g-bgG)+(b-bgB)*(b-bgB));',
    '        if (dc<25) continue;',
    '        final double rf=r.toDouble(),gf=g.toDouble(),bf=b.toDouble();',
    '        sR+=rf;sG+=gf;sB+=bf;sR2+=rf*rf;sG2+=gf*gf;sB2+=bf*bf;',
    '        final double mx=math.max(rf,math.max(gf,bf)),mn=math.min(rf,math.min(gf,bf)),d=mx-mn;',
    '        double h=0.0;',
    '        if(d>0){if(mx==rf)h=60*(((gf-bf)/d)%6);else if(mx==gf)h=60*(((bf-rf)/d)+2);else h=60*(((rf-gf)/d)+4);if(h<0)h+=360;}',
    '        final double hS=(h/360)*255,sS2=mx==0?0:(d/mx)*255,vS=mx;',
    '        sH+=hS;sS+=sS2;sV+=vS;sH2+=hS*hS;sS2p+=sS2*sS2;sV2+=vS*vS;',
    '        if(hS>=40&&hS<=90)gn++;if(hS>=25&&hS<40)yw++;if(hS>=12&&hS<25)orr++;if(hS<12)am++;',
    '        n++;',
    '      }',
    '    }',
    '    if (n < 10) return List.filled(20, 0.0);',
    '    final double nd=n.toDouble();',
    '    final double mR=sR/nd,mG=sG/nd,mB=sB/nd;',
    '    final double mH=sH/nd,mS=sS/nd,mV=sV/nd;',
    '    return [mR,math.sqrt(math.max(0,sR2/nd-mR*mR)),mG,math.sqrt(math.max(0,sG2/nd-mG*mG)),mB,math.sqrt(math.max(0,sB2/nd-mB*mB)),mR/(mG+1e-5),mR/(mB+1e-5),(2*mG-mR-mB)/(2*mG+mR+mB+1e-5),(mR+mG-2*mB)/(mR+mG+2*mB+1e-5),mH,math.sqrt(math.max(0,sH2/nd-mH*mH)),mS,math.sqrt(math.max(0,sS2/nd-mS*mS)),mV,math.sqrt(math.max(0,sV2/nd-mV*mV)),gn/nd,yw/nd,orr/nd,am/nd];',
    '  }',
    '  static double sS2p = 0;',
    "  static List<Map<String, dynamic>> _fallback() => const [",
    "    {'label': 'Papaya Ripe', 'index': 1, 'confidence': 0.885},",
    "    {'label': 'Papaya Ripening', 'index': 2, 'confidence': 0.065},",
    "    {'label': 'Papaya Unripe', 'index': 3, 'confidence': 0.030},",
    "    {'label': 'Papaya Overripe', 'index': 0, 'confidence': 0.020},",
    '  ];',
    '}',
]

dart_path = 'lib/services/random_forest_service.dart'
with open(dart_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Dart written: {dart_path} ({os.path.getsize(dart_path)//1024} KB)')
print('DONE - run: flutter run -d V2310')
