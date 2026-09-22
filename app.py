import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app=FastAPI(title="PS45 Electricity Analytics",version="1.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
STATE={"df":None,"an":None}

def make_data(days=450):
    rng=np.random.default_rng(42)
    ts=pd.date_range("2025-07-01",periods=days*24,freq="h")
    areas=["Anna Nagar","T. Nagar","Velachery","Adyar","Ambattur"]
    cats=["Residential","Commercial","Industrial"]
    d=pd.DataFrame([(t,a,c) for t in ts for a in areas for c in cats],columns=["timestamp","area","category"])
    h=d.timestamp.dt.hour.to_numpy(); m=d.timestamp.dt.month.to_numpy(); w=(d.timestamp.dt.dayofweek.to_numpy()>=5)
    base=d.category.map({"Residential":40,"Commercial":65,"Industrial":110}).to_numpy(float)
    mult=d.area.map({"Anna Nagar":1.15,"T. Nagar":1.30,"Velachery":1.0,"Adyar":1.05,"Ambattur":.90}).to_numpy(float)
    res=.55+.35*np.exp(-((h-7)**2)/6)+.9*np.exp(-((h-20)**2)/8)
    com=np.where((h>=7)&(h<=21),.35+.95*np.exp(-((h-13)**2)/30),.25)
    ind=.85+.15*np.sin(h/24*2*np.pi)
    shape=np.select([d.category.eq("Residential"),d.category.eq("Commercial")],[res,com],default=ind)
    season=np.select([np.isin(m,[12,1,2]),np.isin(m,[3,4,5]),np.isin(m,[6,7,8,9])],[.90,1.35,1.05],default=1.0)
    weekend=np.where(d.category.eq("Residential"),np.where(w,1.10,1.0),np.where(w,.55,1.0))
    names=np.select([np.isin(m,[12,1,2]),np.isin(m,[3,4,5]),np.isin(m,[6,7,8,9])],["Winter","Summer","Monsoon"],default="Post-Monsoon")
    tb=np.select([np.isin(m,[12,1,2]),np.isin(m,[3,4,5]),np.isin(m,[6,7,8,9])],[24,34,28],default=27)
    temp=np.round(tb+4*np.sin((h-9)/24*2*np.pi)+rng.normal(0,1.2,len(d)),1)
    y=base*mult*shape*season*weekend*(1+np.maximum(0,temp-28)*.02)*rng.normal(1,.06,len(d))
    y=np.maximum(.5,y); an=rng.random(len(d))<.004; spike=an&(rng.random(len(d))<.6); y[spike]*=rng.uniform(2.2,3.5,spike.sum()); y[an&~spike]*=rng.uniform(.05,.25,(an&~spike).sum())
    d["consumption_kwh"]=np.round(y,2); d["temperature_c"]=temp; d["season"]=names
    return d

def load(raw=None):
    if raw is None: return make_data()
    d=pd.read_csv(io.BytesIO(raw)); req={"timestamp","area","category","consumption_kwh"}; miss=req-set(d.columns)
    if miss: raise ValueError("Missing columns: "+", ".join(sorted(miss)))
    d["timestamp"]=pd.to_datetime(d.timestamp,errors="coerce"); d=d.dropna(subset=["timestamp"]).copy()
    d["consumption_kwh"]=pd.to_numeric(d.consumption_kwh,errors="coerce")
    d["consumption_kwh"]=d["consumption_kwh"].fillna(d.groupby("category")["consumption_kwh"].transform("median"))
    d=d.drop_duplicates(subset=["timestamp","area","category"]); d=d[d.consumption_kwh>=0].copy()
    if "season" not in d: d["season"]=d.timestamp.dt.month.map(lambda x:"Winter" if x in [12,1,2] else "Summer" if x in [3,4,5] else "Monsoon" if x in [6,7,8,9] else "Post-Monsoon")
    if "temperature_c" not in d: d["temperature_c"]=np.nan
    return d.reset_index(drop=True)

def filt(d,area=None,category=None,start=None,end=None):
    x=d
    if area and area!="All": x=x[x.area==area]
    if category and category!="All": x=x[x.category==category]
    if start: x=x[x.timestamp>=pd.Timestamp(start)]
    if end: x=x[x.timestamp<=pd.Timestamp(end)+pd.Timedelta(days=1)]
    return x

def anomaly(d):
    x=d[["consumption_kwh","temperature_c"]].copy(); x["temperature_c"]=x.temperature_c.fillna(0)
    z=IsolationForest(n_estimators=80,contamination=.01,random_state=42,n_jobs=-1).fit(x)
    o=d.copy(); o["is_anomaly"]=z.predict(x)==-1; o["anomaly_score"]=-z.score_samples(x); return o

@app.on_event("startup")
def startup():
    STATE["df"]=make_data(); STATE["an"]=anomaly(STATE["df"])

@app.get("/",response_class=HTMLResponse)
def home(): return HTML

@app.get("/api/filters")
def filters():
    d=STATE["df"]; return {"areas":["All"]+sorted(d.area.unique()),"categories":["All"]+sorted(d.category.unique()),"date_min":str(d.timestamp.min().date()),"date_max":str(d.timestamp.max().date())}

@app.get("/api/kpis")
def kpis(area=None,category=None,start=None,end=None):
    d=filt(STATE["df"],area,category,start,end)
    if d.empty: raise HTTPException(404,"No data")
    p=d.loc[d.consumption_kwh.idxmax()]
    return {"total":round(d.consumption_kwh.sum(),1),"average":round(d.consumption_kwh.mean(),2),"maximum":round(d.consumption_kwh.max(),2),"peak_time":p.timestamp.strftime("%Y-%m-%d %H:%M"),"records":len(d)}

@app.get("/api/area")
def area(category=None):
    g=filt(STATE["df"],category=category).groupby("area").consumption_kwh.sum().sort_values(ascending=False)
    return [{"label":str(k),"value":round(v,1)} for k,v in g.items()]

@app.get("/api/category")
def category(area=None):
    g=filt(STATE["df"],area=area).groupby("category").consumption_kwh.sum().sort_values(ascending=False)
    return [{"label":str(k),"value":round(v,1)} for k,v in g.items()]

@app.get("/api/hourly")
def hourly(area=None,category=None):
    d=filt(STATE["df"],area,category); g=d.groupby(d.timestamp.dt.hour).consumption_kwh.mean()
    return [{"label":int(k),"value":round(v,1)} for k,v in g.items()]

@app.get("/api/monthly")
def monthly(area=None,category=None):
    d=filt(STATE["df"],area,category); g=d.groupby(d.timestamp.dt.to_period("M")).consumption_kwh.sum()
    return [{"label":str(k),"value":round(v,1)} for k,v in g.items()]

@app.get("/api/forecast")
def forecast(area=None,category=None,horizon=14):
    d=filt(STATE["df"],area,category).copy(); d=d.groupby(d.timestamp.dt.date).consumption_kwh.sum().reset_index(); d.columns=["date","y"]; d.date=pd.to_datetime(d.date)
    for lag in [1,2,3,7,14]: d["l"+str(lag)]=d.y.shift(lag)
    d["dow"]=d.date.dt.dayofweek; d["month"]=d.date.dt.month; d=d.dropna(); f=["dow","month","l1","l2","l3","l7","l14"]
    if len(d)<60: raise HTTPException(400,"Not enough data")
    split=len(d)-30; model=RandomForestRegressor(n_estimators=120,random_state=42,n_jobs=-1,min_samples_leaf=2); model.fit(d.iloc[:split][f],d.iloc[:split].y)
    p=model.predict(d.iloc[split:][f]); mae=mean_absolute_error(d.iloc[split:].y,p); rmse=mean_squared_error(d.iloc[split:].y,p)**.5; r2=r2_score(d.iloc[split:].y,p)
    vals=list(d.y.tail(14)); last=d.date.max(); out=[]
    for i in range(int(horizon)):
        nd=last+pd.Timedelta(days=i+1); row={"dow":nd.dayofweek,"month":nd.month}
        for lag in [1,2,3,7,14]: row["l"+str(lag)]=vals[-lag]
        y=float(model.predict(pd.DataFrame([row])[f])[0]); vals.append(y); out.append({"date":str(nd.date()),"value":round(y,1)})
    return {"metrics":{"mae":round(mae,2),"rmse":round(rmse,2),"r2":round(r2,3)},"future":out}

@app.get("/api/anomalies")
def anomalies(area=None,category=None):
    d=filt(STATE["an"],area,category); x=d[d.is_anomaly].nlargest(20,"anomaly_score")
    return [{"timestamp":r.timestamp.strftime("%Y-%m-%d %H:%M"),"area":r.area,"category":r.category,"value":round(r.consumption_kwh,1)} for r in x.itertuples()]

@app.get("/api/recommendations")
def recommendations():
    d=STATE["df"]; a=d.groupby("area").consumption_kwh.sum().idxmax(); c=d.groupby("category").consumption_kwh.sum().idxmax(); h=int(d.groupby(d.timestamp.dt.hour).consumption_kwh.mean().idxmax()); n=int(STATE["an"].is_anomaly.sum())
    return [{"title":"Target the highest-use area","detail":f"{a} has the highest total consumption.","priority":"Medium"},{"title":"Plan for peak hours","detail":f"Average demand is highest around {h:02d}:00.","priority":"High"},{"title":"Review abnormal records","detail":f"{n:,} records are flagged by the anomaly detector.","priority":"High"},{"title":"Focus efficiency programs","detail":f"{c} is the largest consumer category.","priority":"Medium"}]

@app.post("/api/upload")
async def upload(file:UploadFile=File(...)):
    raw=await file.read()
    try: STATE["df"]=load(raw); STATE["an"]=anomaly(STATE["df"])
    except Exception as e: raise HTTPException(400,str(e))
    return {"status":"ok","rows":len(STATE["df"])}

@app.get("/api/export/anomalies.csv")
def export():
    x=STATE["an"][STATE["an"].is_anomaly][["timestamp","area","category","consumption_kwh","anomaly_score"]]; b=io.StringIO(); x.to_csv(b,index=False)
    return StreamingResponse(iter([b.getvalue()]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=anomalies.csv"})

HTML="""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PS45 Electricity Analytics</title><style>
body{margin:0;background:#07101f;color:#eef3fb;font:14px Arial,sans-serif}header{padding:24px 30px;border-bottom:1px solid #263650;display:flex;justify-content:space-between;gap:20px;align-items:center;flex-wrap:wrap}.brand h1{margin:0 0 6px;font-size:28px}.muted{color:#91a0b8}.wrap{max-width:1400px;margin:auto;padding:22px}.toolbar{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:16px}select,input,button{background:#111d31;color:#eef3fb;border:1px solid #334764;padding:10px 12px;border-radius:7px}button{cursor:pointer}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.card,.panel{background:#0f1a2d;border:1px solid #263650;border-radius:10px;padding:16px}.label{color:#91a0b8}.value{display:block;font-size:24px;font-weight:700;margin:8px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.wide{grid-column:1/-1}.panel h3{margin:0 0 14px}.chart{height:250px;display:flex;align-items:end;gap:7px;padding:12px 5px 28px;border-bottom:1px solid #263650;overflow:hidden}.bar{min-width:18px;flex:1;background:#4e78a8;border-radius:4px 4px 0 0;position:relative}.bar span{position:absolute;bottom:-22px;left:50%;transform:translateX(-50%);font-size:10px;color:#91a0b8;white-space:nowrap}.bar em{position:absolute;top:-17px;left:50%;transform:translateX(-50%);font-style:normal;font-size:9px;color:#b9c6d9}.rec{padding:12px 0;border-bottom:1px solid #263650}.rec:last-child{border:0}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #263650;text-align:left;font-size:12px}.scroll{max-height:300px;overflow:auto}.status{margin-bottom:14px;padding:10px 12px;border-radius:7px;background:#10223a;color:#a9c8ee;display:none}.status.show{display:block}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.wide{grid-column:auto}}@media(max-width:600px){.cards{grid-template-columns:1fr}}</style></head><body>
<header><div class="brand"><h1>⚡ PS45 Electricity Analytics</h1><div class="muted">Smart City • Consumption Analytics • Demand Prediction • Anomaly Detection</div></div><div><input id="file" type="file" accept=".csv"><button id="uploadBtn">Upload CSV</button><button id="exportBtn">Export anomalies</button></div></header>
<main class="wrap"><div id="status" class="status"></div><div class="toolbar"><select id="area"></select><select id="cat"></select><input id="start" type="date"><input id="end" type="date"><button id="applyBtn">Apply filters</button></div>
<div class="cards"><div class="card"><span class="label">Total Consumption</span><b class="value" id="total">—</b><span class="muted">kWh</span></div><div class="card"><span class="label">Average</span><b class="value" id="avg">—</b><span class="muted">kWh / record</span></div><div class="card"><span class="label">Peak</span><b class="value" id="max">—</b><span class="muted">kWh</span></div><div class="card"><span class="label">Peak Time</span><b class="value" id="peak" style="font-size:17px">—</b></div><div class="card"><span class="label">Records</span><b class="value" id="rows">—</b></div></div>
<div class="grid"><div class="panel"><h3>Area Consumption</h3><div id="areaChart" class="chart"></div></div><div class="panel"><h3>Category Consumption</h3><div id="catChart" class="chart"></div></div><div class="panel"><h3>Hourly Consumption Profile</h3><div id="hourChart" class="chart"></div></div><div class="panel"><h3>Monthly Consumption Trend</h3><div id="monthChart" class="chart"></div></div><div class="panel wide"><h3>Future Demand Forecast</h3><div id="metrics" class="muted"></div><div id="forecastChart" class="chart"></div></div><div class="panel"><h3>Recommendations</h3><div id="recs"></div></div><div class="panel"><h3>Anomalies Detected</h3><div class="scroll"><table><thead><tr><th>Time</th><th>Area</th><th>Category</th><th>kWh</th></tr></thead><tbody id="an"></tbody></table></div></div></div></main>
<script>
(function(){const $=id=>document.getElementById(id);function msg(t){const s=$("status");s.textContent=t;s.classList.add("show")}async function api(u){const r=await fetch(u);if(!r.ok)throw Error(await r.text());return r.json()}function qs(){return "area="+encodeURIComponent($("area").value)+"&category="+encodeURIComponent($("cat").value)}function draw(id,data,labelKey,valueKey){const el=$(id);el.innerHTML="";if(!data||!data.length){el.textContent="No data";return}const max=Math.max(...data.map(x=>Number(x[valueKey])||0),1);data.forEach(x=>{const b=document.createElement("div");b.className="bar";b.style.height=Math.max(4,(Number(x[valueKey])||0)/max*88)+"%";const v=document.createElement("em");v.textContent=Number(x[valueKey]).toLocaleString();const l=document.createElement("span");l.textContent=x[labelKey];b.append(v,l);el.appendChild(b)})}async function refresh(){try{msg("Loading dashboard...");const minDate=$("start").value,maxDate=$("end").value;if(minDate&&maxDate&&minDate>maxDate)throw Error("Start date cannot be after end date.");const q=qs()+"&start="+minDate+"&end="+maxDate;const k=await api("/api/kpis?"+q);$("total").textContent=k.total.toLocaleString();$("avg").textContent=k.average;$("max").textContent=k.maximum;$("peak").textContent=k.peak_time;$("rows").textContent=k.records.toLocaleString();const [a,c,h,m,f,an,r]=await Promise.all([api("/api/area?category="+encodeURIComponent($("cat").value)),api("/api/category?area="+encodeURIComponent($("area").value)),api("/api/hourly?"+qs()),api("/api/monthly?"+qs()),api("/api/forecast?"+qs()),api("/api/anomalies?"+qs()),api("/api/recommendations")]);draw("areaChart",a,"label","value");draw("catChart",c,"label","value");draw("hourChart",h,"label","value");draw("monthChart",m,"label","value");draw("forecastChart",f.future,"date","value");$("metrics").textContent="Model validation - MAE: "+f.metrics.mae+" • RMSE: "+f.metrics.rmse+" • R²: "+f.metrics.r2;$("recs").innerHTML=r.map(x=>"<div class='rec'><b>"+x.title+"</b><br><span class='muted'>"+x.detail+" • Priority: "+x.priority+"</span></div>").join("");$("an").innerHTML=an.length?an.map(x=>"<tr><td>"+x.timestamp+"</td><td>"+x.area+"</td><td>"+x.category+"</td><td>"+x.value+"</td></tr>").join(""):"<tr><td colspan='4'>No anomalies found</td></tr>";$("status").classList.remove("show")}catch(e){msg("Dashboard error: "+e.message);console.error(e)}}async function init(){try{const f=await api("/api/filters");f.areas.forEach(x=>$("area").add(new Option(x,x)));f.categories.forEach(x=>$("cat").add(new Option(x,x)));$("area").value="All";$("cat").value="All";$("start").value=f.date_min;$("end").value=f.date_max;await refresh()}catch(e){msg("Could not load dashboard data: "+e.message)}}$("applyBtn").addEventListener("click",refresh);$("exportBtn").addEventListener("click",()=>location.href="/api/export/anomalies.csv");$("uploadBtn").addEventListener("click",async()=>{const f=$("file").files[0];if(!f)return msg("Please choose a CSV file first.");const d=new FormData();d.append("file",f);try{const r=await fetch("/api/upload",{method:"POST",body:d});if(!r.ok)throw Error(await r.text());msg("Dataset uploaded successfully.");await init()}catch(e){msg("Upload failed: "+e.message)}});init()})();
</script></body></html>"""
