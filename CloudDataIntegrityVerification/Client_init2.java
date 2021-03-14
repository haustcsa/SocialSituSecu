package jPBC_3;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import it.unisa.dia.gas.jpbc.Element;
import it.unisa.dia.gas.jpbc.Pairing;
import it.unisa.dia.gas.plaf.jpbc.pairing.PairingFactory;

public class Client_init2 {
	private Pairing pairing;
	private Element u,g,x,y;
	private Element TiElement;
	private Element HFiElement;
	private Element uFiElement;
	private Element T;
	private Element P;
	private Element Vi;
	private Element leftResultElement,rightResultElement,rightResultHF;
	long l;
	public void init() {
		this.pairing = PairingFactory.getPairing("a.properties");
		this.u = pairing.getG1().newRandomElement().getImmutable();
		this.g = pairing.getG1().newRandomElement().getImmutable();
		this.x = pairing.getZr().newRandomElement().getImmutable();
		this.y = this.g.powZn(this.x).getImmutable();
	}
	public void creatTi(){
		List<String> fileMD5list = Utils.mD5Util(1, 1);
		TiElement =pairing.getG1().newOneElement();
		HFiElement = pairing.getG1().newOneElement();
		uFiElement = pairing.getG1().newOneElement();
		T = pairing.getG1().newOneElement();
		P = pairing.getZr().newZeroElement();
		System.out.println("T:"+T);
		System.out.println("P:"+P);
		for (int i = 1; i <= fileMD5list.size(); i++) {
			HFiElement = pairing.getG1().newOneElement().setFromHash(fileMD5list.get(i-1).getBytes(), 0, fileMD5list.get(i-1).length()).getImmutable();
			l = (int)i;
			uFiElement=this.u.pow(BigInteger.valueOf(l)).getImmutable();
			TiElement = (HFiElement.mul(uFiElement).getImmutable()).powZn(this.x).getImmutable();
		}
		
		
			Vi = pairing.getZr().newRandomElement().getImmutable();
			T = T.mul(TiElement.getImmutable().powZn(Vi.getImmutable()).getImmutable());
			P =Vi.mul(BigInteger.valueOf(l));
			//P =P.add(Vi.getImmutable().mul(BigInteger.valueOf(l)).getImmutable());
			//rightResultHF = rightResultHF.mul(TiElement.getImmutable().powZn(Vi.getImmutable())).getImmutable();
			//System.out.println((i+1)+"号文件取值2HFiElement:"+HFiListElement.get(i));
			System.out.println("第1回T的值"+T);
			System.out.println("第1回P的值"+P);
			System.out.println("第1回Vi的值"+Vi);
			System.out.println("第1回BigInteger.valueOf(l)的值"+BigInteger.valueOf(l));
	    leftResultElement = pairing.pairing(T, g);
	    rightResultElement = pairing.pairing(HFiElement.getImmutable().powZn(Vi.getImmutable()).duplicate().mul(this.u.powZn(P).getImmutable()).getImmutable(), this.y);
	    System.out.println("leftResultElement:"+leftResultElement);
		System.out.println("rightResultElement"+rightResultElement);
		System.out.println(leftResultElement.equals(rightResultElement));
	}
}
