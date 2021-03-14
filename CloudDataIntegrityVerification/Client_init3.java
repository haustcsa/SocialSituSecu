package jPBC_3;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import it.unisa.dia.gas.jpbc.Element;
import it.unisa.dia.gas.jpbc.Pairing;
import it.unisa.dia.gas.plaf.jpbc.pairing.PairingFactory;

public class Client_init3 {
	private Pairing pairing;
	private Element u,g,x,y;
	private Element TiElement;
	//private Element HFiElement;
	//private Element uFiElement;
	private Element T;
	private Element P;
	private Element tElement;
	private Element pElement;
	private Element leftResultElement,rightResultElement,rightResultHF;
	private List<Element> TiListElement;
	private List<Element> HFiListElement;
	private List<Element> UFiListElement;
	private HashMap<Integer,Element> ViMapElement;
	long l;
	public void init() {
		this.pairing = PairingFactory.getPairing("a.properties");
		//this.u = pairing.getG1().newOneElement().getImmutable();//u取固定值1
		this.u = pairing.getG1().newRandomElement().getImmutable();
		//this.g = pairing.getG1().newOneElement().getImmutable();//g取固定值1
		this.g = pairing.getG1().newRandomElement().getImmutable();
		//this.x = pairing.getZr().newOneElement().getImmutable();//x取固定值1
		this.x = pairing.getZr().newRandomElement().getImmutable();
		this.y = this.g.powZn(this.x).getImmutable();
	}
	public void creatTi(){
		List<String> fileMD5list = Utils.mD5Util(4, 7);
		TiElement =pairing.getG1().newOneElement();
		//HFiElement = pairing.getG1().newOneElement();
		//uFiElement = pairing.getG1().newOneElement();
		T = pairing.getG1().newOneElement();//
		P = pairing.getZr().newZeroElement();// 
		rightResultHF = pairing.getG1().newOneElement();
		TiListElement = new ArrayList<>();
		HFiListElement = new ArrayList<>();
		UFiListElement = new ArrayList<>();
		ViMapElement = new HashMap<>();
		System.out.println("T:"+T);
		System.out.println("P:"+P);
		for (int i = 1; i <= fileMD5list.size(); i++) {
			Element HFiElement = pairing.getG1().newOneElement().setFromHash(fileMD5list.get(i-1).getBytes(), 0, fileMD5list.get(i-1).length()).getImmutable();
			//System.out.println("HFiElement值:"+HFiElement);
			HFiListElement.add(HFiElement.duplicate());
			l = (int)i;
			Element uFiElement=this.u.pow(BigInteger.valueOf(l)).getImmutable();
			TiElement = (HFiElement.mul(uFiElement).getImmutable()).powZn(this.x).getImmutable();
			TiListElement.add(TiElement.duplicate());
			UFiListElement.add(uFiElement.duplicate());
			//System.out.println((i+1)+"号文件传值TiElement:"+TiElement);
			//System.out.println((i+1)+"号文件传值HFiElement:"+HFiElement.duplicate());
			//System.out.println((i+1)+"号文件传值uFiElement:"+uFiElement);
			
		}
	
		for (int i = 1; i <= TiListElement.size(); i++) {
			//Element Vi = pairing.getZr().newOneElement().getImmutable();//Vi定为1
			//System.out.println("Vi值:"+Vi);
			l = (int)i;
			Element Vi = pairing.getZr().newRandomElement().getImmutable();
			//System.out.println((i+1)+"号文件取值UFiElement:"+UFiListElement.get(i));
			//System.out.println((i+1)+"号文件取值HFiElement:"+HFiListElement.get(i));
			//System.out.println((i)+"号文件取值TiElement:"+TiListElement.get(i-1));
			tElement = TiListElement.get(i-1).powZn(Vi);
			T.mul(tElement).getImmutable();
			//T = T.mul(tElement).getImmutable();
			//T = (T.duplicate().mul(TiListElement.get(i-1).getImmutable().powZn(Vi.getImmutable())).getImmutable()).duplicate();		
			//System.out.println("BigInteger.valueOf(l)值:"+BigInteger.valueOf(l));
			pElement = Vi.mul(BigInteger.valueOf(l));
			//System.out.println("P值:"+P);
			//System.out.println("pElement 值:"+pElement );
			P.add(pElement);
			//P = P.add(pElement);
			//P = (P.duplicate().add(Vi.getImmutable().mul(BigInteger.valueOf(l))).getImmutable()).duplicate();
			System.out.println("第"+(i)+"回T的值"+T);
			System.out.println("第"+(i)+"回P的值"+P);
			ViMapElement.put(i, Vi.getImmutable());
			rightResultHF.mul(HFiListElement.get(i-1).powZn(Vi));
			//rightResultHF = rightResultHF.mul(HFiListElement.get(i-1).powZn(Vi));
			//System.out.println((i+1)+"号文件取值2TiiElement:"+TiListElement.get(i));
			//System.out.println((i+1)+"号文件取值2HFiElement:"+HFiListElement.get(i));
		}
		T.getImmutable();
		P.getImmutable();
	    leftResultElement = pairing.pairing(T, g);
	    rightResultElement = pairing.pairing(rightResultHF.mul(this.u.powZn(P)), this.y);
	    System.out.println("leftResultElement:"+leftResultElement);
		System.out.println("rightResultElement"+rightResultElement);
		System.out.println(leftResultElement.equals(rightResultElement));
	}
}
